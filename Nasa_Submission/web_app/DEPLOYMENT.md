# 🚀 NASA TEMPO AQI - Deployment Guide

Multiple deployment options for your hackathon presentation!

## 🎯 Quick Deploy Options

### 1. 🏠 Local Network (Recommended for Hackathon)
**Perfect for presenting to judges on the same network**

```powershell
# Run in PowerShell
.\deploy_local.ps1
```

**Features:**
- ✅ Instant deployment
- ✅ Works on local network 
- ✅ Share with judges: `http://YOUR-IP:8000`
- ✅ Mobile-friendly for judge's phones

---

### 2. ☁️ Cloud Deployment

#### Heroku (Free Tier Available)
```powershell
# Install Heroku CLI first: https://devcenter.heroku.com/articles/heroku-cli
.\deploy_heroku.ps1
```

#### Railway (Modern Alternative)
```powershell
# Install Railway CLI: npm install -g @railway/cli
.\deploy_railway.ps1
```

---

### 3. 🐳 Docker Deployment

```bash
# Build and run with Docker
docker-compose up --build

# Or manual Docker
docker build -t nasa-tempo-aqi .
docker run -p 8000:8000 nasa-tempo-aqi
```

---

## 🎪 Hackathon Presentation Setup

### Option A: Local Network Demo
1. Run `deploy_local.ps1`
2. Share the network URL with judges
3. Demo works on any device connected to the same WiFi

### Option B: Live Cloud Demo
1. Deploy to Heroku/Railway before presentation
2. Get public URL (e.g., `https://your-app.herokuapp.com`)
3. Demo works from anywhere with internet

---

## 🔧 Technical Details

**Dependencies:** Pure Python (no external packages required)
**Port:** 8000 (configurable)
**Browser Support:** All modern browsers
**Mobile:** Fully responsive design

---

## 📊 Features Ready for Demo

✅ **16 North American Cities**
✅ **Realistic AQI Values** (20-50 range)
✅ **EPA-Compliant Color Coding**
✅ **24-Hour Forecasting**
✅ **Pollutant Breakdown**
✅ **Health Recommendations**
✅ **NASA TEMPO Branding**
✅ **Mobile Responsive**
✅ **Real-time Updates**

---

## 🏆 Presentation Tips

1. **Start with local demo** - most reliable
2. **Have cloud backup** - in case of network issues
3. **Test on mobile** - judges love mobile demos
4. **Show different cities** - demonstrates universality
5. **Highlight realistic AQI values** - shows accuracy

**🎯 Your NASA TEMPO AQI app is deployment-ready for an impressive hackathon presentation!**