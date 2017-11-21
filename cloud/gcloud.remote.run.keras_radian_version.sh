export BUCKET_NAME=hand-data
export JOB_NAME="test1_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=europe-west1

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir gs://$BUCKET_NAME/$JOB_NAME \
  --runtime-version 1.0 \
  --module-name trainer.multires_gcloud_radian \
  --package-path ./trainer \
  --region $REGION \
  --config=trainer/cloudml-gpu.yaml \
  -- \
  --train-files gs://hand-data