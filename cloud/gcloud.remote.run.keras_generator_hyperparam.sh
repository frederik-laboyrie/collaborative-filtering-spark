export BUCKET_NAME=hand-data
export JOB_NAME="squeeze_hyper$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=europe-west1
export HPTUNING_CONFIG=hptuning_config.yaml
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir gs://$BUCKET_NAME/$JOB_NAME \
    --runtime-version 1.0 \
    --config $HPTUNING_CONFIG \
    --module-name trainer.multires_gcloud_main_generator_hyperparam \
    --package-path ./trainer \
    --region $REGION \
    -- \
    --train-files gs://hand-data \
    --squeeze_param 16
