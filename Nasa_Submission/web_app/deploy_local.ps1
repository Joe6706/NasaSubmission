# NASA TEMPO AQI - Local Network Deployment Script
# Run this script to deploy on your local network

param(
    [string]$Port = "8000",
    [switch]$OpenBrowser = $true
)

Write-Host "🏠 NASA TEMPO AQI - Local Network Deployment" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Get local IP address
$localIP = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Wi-Fi" | Where-Object {$_.IPAddress -like "192.168.*" -or $_.IPAddress -like "10.*" -or $_.IPAddress -like "172.*"}).IPAddress
if (!$localIP) {
    $localIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -ne "127.0.0.1" -and $_.IPAddress -like "192.168.*"}).IPAddress | Select-Object -First 1
}

if (!$localIP) {
    $localIP = "localhost"
    Write-Host "⚠️ Could not detect local IP. Using localhost" -ForegroundColor Yellow
} else {
    Write-Host "🌐 Local IP detected: $localIP" -ForegroundColor Green
}

# Check if port is available
$portInUse = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "❌ Port $Port is already in use. Trying to stop existing process..." -ForegroundColor Red
    taskkill /F /IM python.exe 2>$null
    Start-Sleep -Seconds 2
}

# Start the server
Write-Host "🚀 Starting NASA TEMPO AQI server..." -ForegroundColor Yellow
Write-Host "📍 Local access: http://localhost:${Port}" -ForegroundColor Cyan
Write-Host "🌐 Network access: http://${localIP}:${Port}" -ForegroundColor Cyan
Write-Host "📱 Mobile access: http://${localIP}:${Port}" -ForegroundColor Cyan
Write-Host "" -ForegroundColor White
Write-Host "🎯 READY FOR HACKATHON DEMO!" -ForegroundColor Green
Write-Host "Share the network URL with judges: http://${localIP}:${Port}" -ForegroundColor Yellow
Write-Host "" -ForegroundColor White
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host "==========================================" -ForegroundColor Cyan

# Open browser if requested
if ($OpenBrowser) {
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:${Port}"
}

# Start the Python server
python server.py