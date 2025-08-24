# 🚀 Automated Deployment Guide

This guide shows how to automatically deploy your Chinese Poetry Generator to Hugging Face Spaces using GitHub Actions.

## 📋 Prerequisites

1. **GitHub Account**: Repository for your code
2. **Hugging Face Account**: For hosting the Spaces app
3. **Fine-tuned Model**: Your `tinyllama-poetry/` LoRA adapter

## 🔧 Setup Steps

### Step 1: Create GitHub Repository

1. Create a new repository on GitHub:
   ```bash
   # Option A: Create on GitHub.com, then clone
   git clone https://github.com/YOUR_USERNAME/chinese-poetry-generator.git
   cd chinese-poetry-generator
   
   # Option B: Initialize locally, then push
   git init
   git remote add origin https://github.com/YOUR_USERNAME/chinese-poetry-generator.git
   ```

### Step 2: Configure GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these **Repository Secrets**:

| Secret Name | Description | Example |
|------------|-------------|---------|
| `HF_TOKEN` | Hugging Face Write Token | `hf_xxxxxxxxxxxx` |
| `HF_USERNAME` | Your HF username | `fmlin429` |
| `SPACE_NAME` | Space name (optional) | `chinese-poetry-generator` |

#### How to get HF_TOKEN:
1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name: `github-actions-deploy`
4. Type: **Write**
5. Copy the token to GitHub secrets

### Step 3: Prepare Your Repository

Your repository should have this structure:
```
chinese-poetry-generator/
├── .github/workflows/deploy-to-hf-spaces.yml  ✅ Auto-deployment
├── .gitignore                                 ✅ Ignore unnecessary files
├── app.py                                     ✅ Gradio web app
├── requirements.txt                           ✅ Dependencies
├── hf_spaces_README.md                        ✅ HF Spaces README
├── training_data.json                         ✅ Your training data
├── tinyllama-poetry/                          ✅ Your LoRA adapter
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
└── DEPLOYMENT_GUIDE.md                        ✅ This guide
```

### Step 4: Push to GitHub

```bash
# Add all files
git add .

# Commit changes
git commit -m "Add Chinese poetry generator with automated deployment"

# Push to GitHub (this triggers deployment!)
git push origin main
```

## 🎯 Automatic Deployment

### When Deployment Triggers:
- ✅ Push to `main` or `master` branch
- ✅ Changes to `app.py`, `requirements.txt`, or `tinyllama-poetry/`
- ✅ Manual trigger via GitHub Actions tab

### What Happens:
1. **Validation**: Checks all required files exist
2. **HF Space Creation**: Creates/updates your Hugging Face Space
3. **File Upload**: Uploads app files and LoRA adapter
4. **Build**: HF Spaces automatically builds and deploys
5. **Success**: Your app is live! 🎉

### Monitor Progress:
- Go to your GitHub repo → **Actions** tab
- Watch the deployment pipeline in real-time
- Check logs if something fails

## 🌐 Access Your App

After successful deployment:
- **URL**: `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`
- **Build time**: ~3-5 minutes for first deployment
- **Updates**: ~1-2 minutes for subsequent deployments

## 🔄 Making Updates

To update your app:
1. **Edit files** locally (app.py, training data, etc.)
2. **Commit and push**:
   ```bash
   git add .
   git commit -m "Update poetry model/UI"
   git push
   ```
3. **Automatic deployment** triggers
4. **App updates** within minutes!

## 🐛 Troubleshooting

### Common Issues:

1. **"HF_TOKEN not found"**
   - Check GitHub Secrets are set correctly
   - Ensure token has **Write** permissions

2. **"Space creation failed"**
   - Verify HF_USERNAME is correct
   - Check space name doesn't conflict

3. **"Model loading failed"**
   - Ensure `tinyllama-poetry/` folder is complete
   - Check adapter files are valid

4. **"Build timeout"**
   - Large model files may cause timeout
   - Try pushing smaller commits

### View Deployment Logs:
1. Go to GitHub repo → **Actions**
2. Click on latest workflow run
3. Expand **"Deploy to Hugging Face Spaces"**
4. Check error messages

### Manual Trigger:
1. Go to GitHub repo → **Actions**
2. Select **"Deploy to Hugging Face Spaces"**
3. Click **"Run workflow"**
4. Choose branch and run

## 🎨 Customization

### Change Space Name:
Update the `SPACE_NAME` secret in GitHub, or modify the workflow file.

### Add Environment Variables:
Add more secrets in GitHub → Settings → Secrets for app configuration.

### Modify Deployment Trigger:
Edit `.github/workflows/deploy-to-hf-spaces.yml` to change when deployment runs.

## 📊 Benefits of This Setup

✅ **Automated**: Push code → App deploys automatically  
✅ **Version Control**: Full history of changes  
✅ **Rollbacks**: Easy to revert to previous versions  
✅ **Collaboration**: Team members can contribute  
✅ **Professional**: CI/CD pipeline like production apps  

## 🎉 Success!

Once set up, you'll have:
- **GitHub repo** with your code
- **Automated deployments** on every push  
- **Public web app** on Hugging Face Spaces
- **Professional workflow** for updates

Your Chinese Poetry Generator is now fully automated! 🏮✨