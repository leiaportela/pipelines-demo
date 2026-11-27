# Project-specific settings for the pipelines demo. This is the only file to edit.
# `source deploy/env.sh` before deploy/setup-gcp.sh, deploy/deploy.sh and the pipeline/*.py commands.
# A value already set in your shell wins over the default here.
export PROJECT=${PROJECT:-your-project-id}
export REGION=${REGION:-us-central1}
export PRESENTER=${PRESENTER:-presenter@example.com}   # who may open the dashboard through IAP

if [ "$PROJECT" = your-project-id ]; then
  echo "deploy/env.sh: PROJECT is still the placeholder; set your project at the top of this file" >&2
fi
