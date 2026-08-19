"""
Resolve the experiment's model lineup to real Bedrock inference profile IDs.

There is no single ID pattern to write down. Newer releases use undated profile
IDs and older ones keep a date stamp and version suffix, and both coexist:

    us.anthropic.claude-sonnet-4-6                   (newer, undated)
    global.anthropic.claude-haiku-4-5-20251001-v1:0  (older, dated)

The geo prefix varies per model too (us. / eu. / apac. / global.). Since the
catalog also moves and availability is per-region, the only reliable source is
your own account. This asks it.

    python experiments/resolve_models.py                    # table + role flags
    python experiments/resolve_models.py --region us-west-2
    python experiments/resolve_models.py --check            # also verify access

Prints the --model-role flags ready to paste, and flags any role that came back
empty or unavailable. Fill the results into aws_setup.md's pinned-config table,
and where an ID is undated record the date you pinned it -- an undated profile
can be repointed at a newer snapshot, so the ID alone does not identify what you
actually ran.
"""
import argparse
import sys

# (role, substrings that must all appear in the id). Substrings rather than
# exact ids on purpose -- the point is to discover the version, not assert it.
LINEUP = [
    ("orchestrator (strong)", ["anthropic", "sonnet"]),
    ("orchestrator (weak)", ["anthropic", "haiku"]),
    ("worker", ["anthropic", "haiku"]),
    ("judge_rubric", ["llama"]),
    ("judge_civ", ["mistral"]),
    ("judge_drb", ["nova"]),
    ("sentinel_node/edge", ["gpt-oss"]),
]

# Roles whose flag is emitted for the eval command. orchestrator (strong) is
# --model, not --model-role, and the weak one is a separate run.
ROLE_FLAGS = ["worker", "judge_rubric", "judge_civ", "judge_drb"]


def candidates(client) -> list:
    """Every addressable text model id: inference profiles first, since most
    current models must be called through one, then bare foundation models for
    the ones that have no profile."""
    ids = []
    paginator = client.get_paginator("list_inference_profiles") \
        if client.can_paginate("list_inference_profiles") else None
    if paginator:
        for page in paginator.paginate():
            ids += [p["inferenceProfileId"] for p in page.get("inferenceProfileSummaries", [])]
    else:
        ids += [p["inferenceProfileId"]
                for p in client.list_inference_profiles().get("inferenceProfileSummaries", [])]

    profile_stems = {i.split(".", 1)[-1] for i in ids}
    for m in client.list_foundation_models(byOutputModality="TEXT").get("modelSummaries", []):
        if m["modelId"] not in profile_stems:
            ids.append(m["modelId"])
    return sorted(set(ids))


GEO_PREFIXES = ("us.", "eu.", "apac.", "jp.", "au.", "global.")


def match(ids: list, needles: list) -> list:
    hits = [i for i in ids if all(n in i.lower() for n in needles)]
    # Prefer cross-region profiles -- a bare model id often is not directly
    # invocable on demand. Which geo prefix a model carries varies per model, so
    # accept any of them rather than ranking one above another.
    return sorted(hits, key=lambda i: (not i.startswith(GEO_PREFIXES), i), reverse=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=None, help="defaults to your configured region")
    ap.add_argument("--check", action="store_true",
                    help="also call GetFoundationModelAvailability on each pick")
    args = ap.parse_args()

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    except ImportError:
        print("pip install boto3", file=sys.stderr)
        return 1

    client = boto3.client("bedrock", **({"region_name": args.region} if args.region else {}))
    region = client.meta.region_name
    print(f"region: {region}\n")

    try:
        ids = candidates(client)
    except NoCredentialsError:
        print("No AWS credentials found. Run `aws configure` or set AWS_ACCESS_KEY_ID / "
              "AWS_SECRET_ACCESS_KEY, then retry.", file=sys.stderr)
        return 1
    except (BotoCoreError, ClientError) as exc:
        print(f"Could not list models: {exc}\n"
              f"Check bedrock:ListInferenceProfiles and bedrock:ListFoundationModels "
              f"are allowed for this principal.", file=sys.stderr)
        return 1

    print(f"{len(ids)} text models addressable in {region}\n")
    picks, missing = {}, []

    for role, needles in LINEUP:
        hits = match(ids, needles)
        if not hits:
            missing.append(role)
            print(f"{role:<24} -- NO MATCH for {needles}")
            continue
        picks[role] = hits[0]
        print(f"{role:<24} -> {hits[0]}")
        for other in hits[1:4]:
            print(f"{'':<24}    also: {other}")
        if len(hits) > 4:
            print(f"{'':<24}    ... and {len(hits) - 4} more")
        print()

    if args.check:
        print("\n--- availability ---")
        for role, model_id in picks.items():
            # This API takes a foundation model id, not a profile id -- a geo
            # prefix makes it raise ValidationException, which reads like the
            # model is missing when it is only being asked about wrongly.
            bare = model_id.split(".", 1)[-1] if model_id.startswith(GEO_PREFIXES) else model_id
            try:
                resp = client.get_foundation_model_availability(modelId=bare)
                status = resp.get("agreementAvailability", {}).get("status", "?")
            except (BotoCoreError, ClientError) as exc:
                status = f"ERROR ({type(exc).__name__})"
            flag = "" if status == "AVAILABLE" else "   <-- not usable yet"
            print(f"{role:<24} {status}{flag}")
        print("NOT_AVAILABLE with authorizationStatus AUTHORIZED usually means the\n"
              "provider's one-time use-case form has not been submitted (step 3).")

    print("\n--- eval command ---")
    print("# CANDIDATES, not decisions. Each pick is the first alphabetical match\n"
          "# for a substring, so it will happily choose an older release over the\n"
          "# one you meant. Diff against the pinned table in aws_setup.md before\n"
          "# running -- that table is the authority, this is a discovery aid.")
    strong = picks.get("orchestrator (strong)", "<unresolved>")
    print("inspect eval experiments/sif_vs_sentinel.py \\")
    print(f"  --model bedrock/{strong} \\")
    for role in ROLE_FLAGS:
        if role in picks:
            print(f"  --model-role {role}=bedrock/{picks[role]} \\")
    sentinel = picks.get("sentinel_node/edge")
    if sentinel:
        print(f"  --model-role sentinel_node=bedrock/{sentinel} \\")
        print(f"  --model-role sentinel_edge=bedrock/{sentinel} \\")
    print("  --max-samples 8")

    weak = picks.get("orchestrator (weak)")
    if weak:
        print(f"\n# capability ablation: rerun with --model bedrock/{weak}")

    if missing:
        print(f"\nUnresolved: {', '.join(missing)}. Either that family is not in "
              f"{region}, or the substring filter in LINEUP needs adjusting -- "
              f"check the full list above before substituting a different family, "
              f"since the role constraints in aws_setup.md still have to hold.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
