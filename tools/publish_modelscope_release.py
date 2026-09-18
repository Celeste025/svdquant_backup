#!/usr/bin/env python3
"""Publish a pre-validated SVDQuant staging release to ModelScope.

The tool deliberately refuses to run without an environment token and an
explicit public-release acknowledgement.  It never reads tokens from files or
prints them.  Run this only after the release owner has reviewed the upstream
licences, especially MiniMax-H3 and FLUX.1-dev.
"""
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path


DATA = Path("/data1/models/svdquant-wjq")
MODEL_REPOS = ("svdquant-rcm-wan2.1-1.3b", "svdquant-minimax-h3", "svdquant-flux1")
DATASET_REPOS = ("svdquant-videoeval-rcm-wan", "svdquant-videoeval-minimax-h3")


def call_supported(fn, **kwargs):
    """Call across ModelScope SDK minor-version signature changes."""
    parameters = inspect.signature(fn).parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return fn(**kwargs)
    return fn(**{key: value for key, value in kwargs.items() if key in parameters})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True, help="ModelScope account or organization")
    parser.add_argument("--release-root", type=Path, default=DATA / "releases/modelscope/v0.1.0")
    parser.add_argument("--only", action="append", choices=MODEL_REPOS + DATASET_REPOS)
    parser.add_argument("--confirm-public-release", action="store_true")
    parser.add_argument("--confirm-h3-community-license", action="store_true")
    parser.add_argument("--confirm-h3-applicable-territory-controls", action="store_true")
    parser.add_argument("--confirm-flux-dev-noncommercial-license", action="store_true")
    parser.add_argument("--no-proxy", action="store_true", help="clear proxy variables before contacting ModelScope")
    parser.add_argument("--private", action="store_true", help="create selected repositories as private, rather than public")
    args = parser.parse_args()
    if args.no_proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ.pop(key, None)
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        raise SystemExit("MODELSCOPE_API_TOKEN must be set in the terminal environment")
    if not args.confirm_public_release:
        raise SystemExit("pass --confirm-public-release after checking all contents")
    selected = set(args.only or (*MODEL_REPOS, *DATASET_REPOS))
    if "svdquant-minimax-h3" in selected and not args.confirm_h3_community_license:
        raise SystemExit("H3 public redistribution requires --confirm-h3-community-license")
    if "svdquant-minimax-h3" in selected and not args.private and not args.confirm_h3_applicable_territory_controls:
        raise SystemExit("H3 must not be published without controls that exclude the EU, UK, Republic of Korea, and US")
    if "svdquant-flux1" in selected and not args.confirm_flux_dev_noncommercial_license:
        raise SystemExit("FLUX.1-dev publication requires --confirm-flux-dev-noncommercial-license")
    try:
        from modelscope.hub.api import HubApi
    except ImportError as error:
        raise SystemExit("install a current `modelscope` SDK in the publishing environment") from error
    api = HubApi()
    call_supported(api.login, access_token=token, token=token)
    for repo in (*MODEL_REPOS, *DATASET_REPOS):
        if repo not in selected:
            continue
        source = args.release_root / ("datasets" if repo in DATASET_REPOS else "") / repo
        if repo in DATASET_REPOS:
            complete = source.is_dir() and (source / "SHA256SUMS").is_file()
        else:
            complete = source.is_dir() and bool(list((source / "variants").glob("*/SHA256SUMS")))
        if not complete:
            raise SystemExit(f"staging package is incomplete: {source}")
        model_id = f"{args.namespace}/{repo}"
        repo_type = "dataset" if repo in DATASET_REPOS else "model"
        license_name = "Apache License 2.0" if repo in {"svdquant-rcm-wan2.1-1.3b", "svdquant-videoeval-rcm-wan"} else "Other"
        description = "Immutable SVDQuant v0.1.0 release; see per-variant artifact manifests and licenses."
        # Use the current SDK's lower-level repository API so visibility and
        # license metadata are explicit rather than relying on upload defaults.
        if not api._api.repo_exists(model_id, repo_type=repo_type):
            api._api.create_repo(model_id, repo_type=repo_type,
                                 visibility="private" if args.private else "public", license=license_name,
                                 description=description)
        # The exact create API changed between SDK releases; upload_folder
        # creates missing repositories on current SDKs, and fails safely if it
        # cannot.  This keeps the local release tooling version-tolerant.
        try:
            call_supported(api.upload_folder, repo_id=model_id, folder_path=str(source), repo_type=repo_type,
                           commit_message="Publish immutable SVDQuant v0.1.0 release", max_workers=4)
        except AttributeError as error:
            raise SystemExit("installed ModelScope SDK lacks HubApi.upload_folder; upgrade modelscope") from error
        print(f"published {repo_type}: {model_id}", flush=True)


if __name__ == "__main__":
    main()
