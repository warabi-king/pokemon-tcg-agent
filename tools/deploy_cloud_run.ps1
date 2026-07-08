param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,
    [string]$Region = "asia-northeast1",
    [string]$Service = "poketcg-duel"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud CLI was not found. Install the Google Cloud CLI first."
}

$secretBytes = New-Object byte[] 48
$random = [Security.Cryptography.RandomNumberGenerator]::Create()
$random.GetBytes($secretBytes)
$random.Dispose()
$flaskSecret = [Convert]::ToBase64String($secretBytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")

gcloud config set project $ProjectId
if ($LASTEXITCODE -ne 0) { throw "Could not select the GCP project." }

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
if ($LASTEXITCODE -ne 0) { throw "Could not enable the required GCP APIs." }

gcloud run deploy $Service `
    --source . `
    --region $Region `
    --allow-unauthenticated `
    --execution-environment gen2 `
    --cpu 1 `
    --memory 2Gi `
    --concurrency 20 `
    --min-instances 0 `
    --max-instances 1 `
    --set-env-vars "FLASK_SECRET_KEY=$flaskSecret"

if ($LASTEXITCODE -ne 0) { throw "Cloud Run deployment failed." }

Write-Host "Deployment completed. Open the Service URL shown above."
