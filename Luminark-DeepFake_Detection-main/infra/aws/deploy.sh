#!/bin/bash
set -e

# AWS Lambda Deployment Script for Luminark Backend
# Usage: ./deploy.sh <aws-account-id> <region>

AWS_ACCOUNT_ID=$1
AWS_REGION=${2:-us-east-1}
REPOSITORY_NAME="luminark-backend"
FUNCTION_NAME="luminark-backend"

if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Usage: ./deploy.sh <aws-account-id> [region]"
    echo "Example: ./deploy.sh 123456789012 us-east-1"
    exit 1
fi

echo "🚀 Deploying Luminark Backend to AWS Lambda"
echo "Account: $AWS_ACCOUNT_ID"
echo "Region: $AWS_REGION"
echo ""

# Step 1: Authenticate to ECR
echo "📦 Step 1: Authenticating to ECR..."
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Step 2: Create ECR repository (if it doesn't exist)
echo "📦 Step 2: Creating ECR repository..."
aws ecr describe-repositories --repository-names $REPOSITORY_NAME --region $AWS_REGION 2>/dev/null || \
    aws ecr create-repository --repository-name $REPOSITORY_NAME --region $AWS_REGION

# Step 3: Build Docker image
echo "🔨 Step 3: Building Docker image..."
docker build -f infra/aws/Dockerfile.lambda -t $REPOSITORY_NAME:latest .

# Step 4: Tag image for ECR
echo "🏷️  Step 4: Tagging image..."
IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:latest"
docker tag $REPOSITORY_NAME:latest $IMAGE_URI

# Step 5: Push to ECR
echo "⬆️  Step 5: Pushing to ECR..."
docker push $IMAGE_URI

# Step 6: Update Lambda function (or create if doesn't exist)
echo "🔄 Step 6: Updating Lambda function..."
aws lambda update-function-code \
    --function-name $FUNCTION_NAME \
    --image-uri $IMAGE_URI \
    --region $AWS_REGION 2>/dev/null || \
    echo "⚠️  Function doesn't exist. Create it manually via AWS Console with:"
    echo "   Image URI: $IMAGE_URI"
    echo "   Memory: 8192 MB"
    echo "   Timeout: 900 seconds"

echo ""
echo "✅ Deployment complete!"
echo "Image URI: $IMAGE_URI"
