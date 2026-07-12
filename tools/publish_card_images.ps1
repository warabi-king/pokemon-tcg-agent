param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,
    [string]$Region = "asia-northeast1",
    [string]$Bucket = ""
)

$ErrorActionPreference = "Stop"
if (-not $Bucket) { $Bucket = "$ProjectId-poketcg-cards" }
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud CLI was not found."
}

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
& $python tools/build_card_thumbnails.py
if ($LASTEXITCODE -ne 0) { throw "Could not generate card thumbnails. Install Pillow first." }

gcloud storage buckets describe "gs://$Bucket" --project $ProjectId 2>$null
if ($LASTEXITCODE -ne 0) {
    gcloud storage buckets create "gs://$Bucket" `
        --project $ProjectId `
        --location $Region `
        --uniform-bucket-level-access
    if ($LASTEXITCODE -ne 0) { throw "Could not create the card image bucket." }
}

gcloud storage buckets add-iam-policy-binding "gs://$Bucket" `
    --member allUsers `
    --role roles/storage.objectViewer
if ($LASTEXITCODE -ne 0) { throw "Could not make card images publicly readable." }

gcloud storage cp "docs/card-assets/cards/*.webp" "gs://$Bucket/cards/" `
    --cache-control "public,max-age=31536000,immutable"
if ($LASTEXITCODE -ne 0) { throw "Card image upload failed." }

Write-Output "https://storage.googleapis.com/$Bucket/cards"
