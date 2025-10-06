# NASA TEMPO AQI - Heroku Deployment Script
# Run this script to deploy to Heroku

Write-Host "🚀 NASA TEMPO AQI - Heroku Deployment" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if Heroku CLI is installed
if (!(Get-Command heroku -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Heroku CLI not found. Please install it from: https://devcenter.heroku.com/articles/heroku-cli" -ForegroundColor Red
    exit 1
}

# Login to Heroku
Write-Host "🔐 Logging into Heroku..." -ForegroundColor Yellow
heroku login

# Create Heroku app
$appName = Read-Host "Enter your Heroku app name (leave blank for auto-generated)"
if ($appName -eq "") {
    Write-Host "🎲 Creating app with auto-generated name..." -ForegroundColor Yellow
    heroku create
} else {
    Write-Host "🎯 Creating app: $appName" -ForegroundColor Yellow
    heroku create $appName
}

# Set environment variables
Write-Host "🔧 Setting environment variables..." -ForegroundColor Yellow
heroku config:set PORT=8000
heroku config:set PYTHONUNBUFFERED=1

# Deploy to Heroku
Write-Host "🚀 Deploying to Heroku..." -ForegroundColor Yellow
git add .
git commit -m "Deploy NASA TEMPO AQI to Heroku"
git push heroku main

# Open the app
Write-Host "🌐 Opening your deployed app..." -ForegroundColor Green
heroku open

Write-Host "✅ Deployment complete!" -ForegroundColor Green
Write-Host "📊 View logs: heroku logs --tail" -ForegroundColor Cyan
Write-Host "🔧 Manage app: https://dashboard.heroku.com/apps" -ForegroundColor Cyan