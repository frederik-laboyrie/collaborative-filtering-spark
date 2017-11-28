export BUCKET_NAME=hand-data
export JOB_NAME="test1_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=europe-west1
export HPTUNING_CONFIG=vanilla_hptuning_config.yaml

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir gs://$BUCKET_NAME/$JOB_NAME \
  --runtime-version 1.0 \
  --module-name multires_gcloud_vanilla_bw \
  --package-path ./trainer \
  --region $REGION \
  --config $HPTUNING_CONFIG \
  -- \
  --train-files gs://hand-data \
  --kernel_size 5 \
  --filters 16 \
  --top_neurons 128 \
  --dropout 0.5