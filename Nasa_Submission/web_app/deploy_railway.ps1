# NASA TEMPO AQI - Railway Deployment Script
# Run this script to deploy to Railway

Write-Host "🚄 NASA TEMPO AQI - Railway Deployment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if Railway CLI is installed
if (!(Get-Command railway -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Railway CLI not found. Installing..." -ForegroundColor Yellow
    npm install -g @railway/cli
}

# Login to Railway
Write-Host "🔐 Logging into Railway..." -ForegroundColor Yellow
railway login

# Initialize Railway project
Write-Host "🎯 Initializing Railway project..." -ForegroundColor Yellow
railway init

# Deploy to Railway
Write-Host "🚀 Deploying to Railway..." -ForegroundColor Yellow
railway up

# Get deployment URL
Write-Host "🌐 Getting deployment URL..." -ForegroundColor Green
$url = railway status
Write-Host "✅ Deployment complete!" -ForegroundColor Green
Write-Host "📊 View logs: railway logs" -ForegroundColor Cyan
Write-Host "🔧 Manage app: https://railway.app/dashboard" -ForegroundColor Cyan