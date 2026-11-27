#!/usr/bin/env bash
# Build and deploy the dashboard from a clean export of one commit (not from your working tree).
# Usage: deploy/deploy.sh [commit-ish]   (default: HEAD of the repository containing this script)
set -euo pipefail

source "$(dirname "$0")/env.sh"
P=$PROJECT
R=$REGION
SVC=vertex-demo
BUILD_SA=vertex-demo-build@$P.iam.gserviceaccount.com
RUN_SA=vertex-demo-run@$P.iam.gserviceaccount.com
BUCKET=$P-vertex-demo-build
REPO=$R-docker.pkg.dev/$P/vertex-demo

repo_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
sha=$(git -C "$repo_root" rev-parse "${1:-HEAD}")
d=/tmp/deploy-$sha
rm -rf "$d" && mkdir -p "$d"
git -C "$repo_root" archive "$sha" | tar -x -C "$d"
image=$REPO/dashboard:$sha

previous=$(gcloud run services describe $SVC --project=$P --region=$R \
  --format='value(status.latestReadyRevisionName)' 2>/dev/null || true)

gcloud builds submit "$d" --project=$P --region=$R \
  --service-account=projects/$P/serviceAccounts/$BUILD_SA \
  --gcs-source-staging-dir=gs://$BUCKET/source \
  --default-buckets-behavior=regional-user-owned-bucket \
  --config="$d/deploy/cloudbuild.yaml" --substitutions=_IMAGE="$image"

gcloud run deploy $SVC --project=$P --region=$R --image="$image" \
  --service-account=$RUN_SA --no-allow-unauthenticated --iap \
  --set-env-vars=BQ_PROJECT=$P,BQ_DATASET=vertex_demo_dashboard \
  --labels=app=vertex-demo --quiet

gcloud run services update-traffic $SVC --project=$P --region=$R --to-latest

current=$(gcloud run services describe $SVC --project=$P --region=$R \
  --format='value(status.latestReadyRevisionName)')
echo "commit:   $sha"
echo "revision: $current"
echo "previous: ${previous:-none}"
