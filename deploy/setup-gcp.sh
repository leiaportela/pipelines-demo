#!/usr/bin/env bash
# Vertex AI Pipelines demo: one-time GCP setup in $PROJECT (from deploy/env.sh). Idempotent.
# Run it as a project owner:
#   DRY_RUN=1 bash deploy/setup-gcp.sh   # print the plan (default)
#   DRY_RUN=0 bash deploy/setup-gcp.sh   # apply
# Touches only vertex-demo* / vertex_demo* resources. Cloud Run access is granted on the service itself, never with a
# project-level condition (conditions on resource.name don't scope Cloud Run).
set -euo pipefail

source "$(dirname "$0")/env.sh"
P=$PROJECT
R=$REGION
BQ_LOC=US                                  # thelook_ecommerce is in the US multi-region
VIEWER=user:$PRESENTER                     # IAP access to the dashboard
BUCKET=$P-vertex-demo                      # pipeline root, staging, trainer packages
BUILD_BUCKET=$P-vertex-demo-build          # Cloud Build source staging
AR_REPO=vertex-demo                        # Artifact Registry (docker): dashboard image
DS=vertex_demo                             # features, labels, BigQuery ML models, predictions
DASH_DS=vertex_demo_dashboard              # the only dataset the dashboard reads
SVC=vertex-demo                            # Cloud Run service (dashboard)
PIPE_SA=vertex-demo-pipelines@$P.iam.gserviceaccount.com
RUN_SA=vertex-demo-run@$P.iam.gserviceaccount.com
DEPLOY_SA=vertex-demo-deployer@$P.iam.gserviceaccount.com
BUILD_SA=vertex-demo-build@$P.iam.gserviceaccount.com
PN=$(gcloud projects describe $P --format='value(projectNumber)')
VERTEX_AGENT=service-$PN@gcp-sa-aiplatform.iam.gserviceaccount.com
IAP_AGENT=service-$PN@gcp-sa-iap.iam.gserviceaccount.com
DRY_RUN=${DRY_RUN:-1}

run() { echo "+ $*"; if [[ "$DRY_RUN" != 1 ]]; then "$@"; fi; }
exists() { "$@" >/dev/null 2>&1; }
# ds_bind MEMBER ROLE DATASET. `bq add-iam-policy-binding --dataset` can need allowlisting, so this adds
# an entry to the dataset's access list through the REST API instead (legacy role names; SAs use userByEmail).
ds_bind() {
  echo "+ dataset access: $2 on $P:$3 for $1"
  [[ "$DRY_RUN" == 1 ]] && return 0
  TOKEN=$(gcloud auth print-access-token) python3 - "$P" "$3" "$1" "$2" <<'PY'
import json, os, sys, urllib.request
project, ds, member, role = sys.argv[1:]
legacy = {"roles/bigquery.dataEditor": "WRITER", "roles/bigquery.dataViewer": "READER"}[role]
email = member.split(":", 1)[1]
url = f"https://bigquery.googleapis.com/bigquery/v2/projects/{project}/datasets/{ds}"
hdr = {"Authorization": "Bearer " + os.environ["TOKEN"], "Content-Type": "application/json"}
acc = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=hdr))).get("access", [])
if any(e.get("userByEmail", "").lower() == email.lower() and e.get("role") == legacy for e in acc):
    print("  already there"); sys.exit(0)
acc.append({"role": legacy, "userByEmail": email})
urllib.request.urlopen(urllib.request.Request(url, data=json.dumps({"access": acc}).encode(), method="PATCH", headers=hdr))
print("  added")
PY
}

echo "== APIs and service agents"
run gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com storage.googleapis.com \
  artifactregistry.googleapis.com cloudbuild.googleapis.com run.googleapis.com iap.googleapis.com \
  cloudresourcemanager.googleapis.com --project=$P
run gcloud beta services identity create --service=aiplatform.googleapis.com --project=$P
run gcloud beta services identity create --service=iap.googleapis.com --project=$P

echo "== service accounts"
for sa in vertex-demo-pipelines:"Vertex demo pipeline runner" \
          vertex-demo-run:"Vertex demo dashboard runtime" \
          vertex-demo-deployer:"Vertex demo deployer (service vertex-demo only)" \
          vertex-demo-build:"Vertex demo Cloud Build"; do
  n=${sa%%:*}
  exists gcloud iam service-accounts describe "$n@$P.iam.gserviceaccount.com" --project=$P \
    || run gcloud iam service-accounts create "$n" --project=$P --display-name="${sa#*:}"
done

echo "== buckets (uniform access, public access prevention)"
for b in $BUCKET $BUILD_BUCKET; do
  exists gcloud storage buckets describe gs://$b \
    || run gcloud storage buckets create gs://$b --project=$P --location=$R \
         --uniform-bucket-level-access --public-access-prevention
done

echo "== BigQuery datasets ($BQ_LOC)"
for d in $DS $DASH_DS; do
  exists bq --project_id=$P show --dataset "$P:$d" \
    || run bq --project_id=$P mk --dataset --location=$BQ_LOC --label=app:vertex-demo "$P:$d"
done

echo "== Artifact Registry"
exists gcloud artifacts repositories describe $AR_REPO --location=$R --project=$P \
  || run gcloud artifacts repositories create $AR_REPO --repository-format=docker --location=$R --project=$P

echo "== pipeline runner: Vertex AI jobs, its two datasets, its bucket"
run gcloud projects add-iam-policy-binding --format=none $P --member=serviceAccount:$PIPE_SA --role=roles/aiplatform.user --condition=None
run gcloud projects add-iam-policy-binding --format=none $P --member=serviceAccount:$PIPE_SA --role=roles/bigquery.jobUser --condition=None
ds_bind serviceAccount:$PIPE_SA roles/bigquery.dataEditor $DS
ds_bind serviceAccount:$PIPE_SA roles/bigquery.dataEditor $DASH_DS
run gcloud storage buckets add-iam-policy-binding --format=none gs://$BUCKET --member=serviceAccount:$PIPE_SA --role=roles/storage.objectAdmin
run gcloud storage buckets add-iam-policy-binding --format=none gs://$BUCKET --member=serviceAccount:$PIPE_SA --role=roles/storage.legacyBucketReader
# pipeline steps launch custom and tuning jobs as this same SA
run gcloud iam service-accounts add-iam-policy-binding --format=none $PIPE_SA --project=$P --member=serviceAccount:$PIPE_SA \
  --role=roles/iam.serviceAccountUser

echo "== Vertex AI service agent: AutoML reads the training table, batch prediction writes results"
ds_bind serviceAccount:$VERTEX_AGENT roles/bigquery.dataEditor $DS

echo "== dashboard runtime: read the dashboard dataset only"
run gcloud projects add-iam-policy-binding --format=none $P --member=serviceAccount:$RUN_SA --role=roles/bigquery.jobUser --condition=None
ds_bind serviceAccount:$RUN_SA roles/bigquery.dataViewer $DASH_DS

echo "== build SA: push images to its repo, read staged source, write logs"
run gcloud artifacts repositories add-iam-policy-binding --format=none $AR_REPO --location=$R --project=$P \
  --member=serviceAccount:$BUILD_SA --role=roles/artifactregistry.writer
run gcloud storage buckets add-iam-policy-binding --format=none gs://$BUILD_BUCKET --member=serviceAccount:$BUILD_SA --role=roles/storage.objectViewer
run gcloud projects add-iam-policy-binding --format=none $P --member=serviceAccount:$BUILD_SA --role=roles/logging.logWriter --condition=None

echo "== deployer SA: builds and deploys the dashboard (Cloud Run rights are on the service, below)"
run gcloud projects add-iam-policy-binding --format=none $P --member=serviceAccount:$DEPLOY_SA --role=roles/cloudbuild.builds.editor --condition=None
run gcloud projects add-iam-policy-binding --format=none $P --member=serviceAccount:$DEPLOY_SA --role=roles/storage.bucketViewer --condition=None
run gcloud projects add-iam-policy-binding --format=none $P --member=serviceAccount:$DEPLOY_SA --role=roles/logging.viewer --condition=None
run gcloud projects add-iam-policy-binding --format=none $P --member=serviceAccount:$DEPLOY_SA --role=roles/serviceusage.serviceUsageConsumer --condition=None
run gcloud storage buckets add-iam-policy-binding --format=none gs://$BUILD_BUCKET --member=serviceAccount:$DEPLOY_SA --role=roles/storage.objectAdmin
run gcloud artifacts repositories add-iam-policy-binding --format=none $AR_REPO --location=$R --project=$P \
  --member=serviceAccount:$DEPLOY_SA --role=roles/artifactregistry.reader
run gcloud iam service-accounts add-iam-policy-binding --format=none $RUN_SA --project=$P --member=serviceAccount:$DEPLOY_SA --role=roles/iam.serviceAccountUser
run gcloud iam service-accounts add-iam-policy-binding --format=none $BUILD_SA --project=$P --member=serviceAccount:$DEPLOY_SA --role=roles/iam.serviceAccountUser

echo "== Cloud Run service: placeholder behind IAP, presenter only"
exists gcloud run services describe $SVC --region=$R --project=$P \
  || run gcloud run deploy $SVC --image=us-docker.pkg.dev/cloudrun/container/hello --region=$R --project=$P \
       --service-account=$RUN_SA --no-allow-unauthenticated --iap --min-instances=0 --max-instances=2 \
       --memory=512Mi --labels=app=vertex-demo --quiet
run gcloud run services add-iam-policy-binding --format=none $SVC --region=$R --project=$P \
  --member=serviceAccount:$DEPLOY_SA --role=roles/run.admin
run gcloud run services add-iam-policy-binding --format=none $SVC --region=$R --project=$P \
  --member=serviceAccount:$IAP_AGENT --role=roles/run.invoker
run gcloud iap web add-iam-policy-binding --format=none --project=$P --resource-type=cloud-run --service=$SVC --region=$R \
  --member=$VIEWER --role=roles/iap.httpsResourceAccessor --condition=None

echo "== done (DRY_RUN=$DRY_RUN)"
echo "Pipeline runs: --service-account=$PIPE_SA, pipeline root gs://$BUCKET/pipeline-root, labels app=vertex-demo."
echo "deploy/deploy.sh then does, roughly:"
echo "  gcloud builds submit --project=$P --region=$R --service-account=projects/$P/serviceAccounts/$BUILD_SA \\"
echo "    --gcs-source-staging-dir=gs://$BUILD_BUCKET/source --default-buckets-behavior=regional-user-owned-bucket \\"
echo "    --tag=$R-docker.pkg.dev/$P/$AR_REPO/dashboard:<sha>"
echo "  gcloud run deploy $SVC --image=... --region=$R --project=$P --service-account=$RUN_SA \\"
echo "    --no-allow-unauthenticated --iap --set-env-vars=BQ_PROJECT=$P,BQ_DATASET=$DASH_DS"
