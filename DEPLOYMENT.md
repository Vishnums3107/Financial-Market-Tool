# Deployment Guide for Stock Dashboard

## Streamlit Cloud Deployment (Recommended)

### Prerequisites
- GitHub account
- Stock dashboard code pushed to GitHub repository

### Steps

1. **Prepare Your Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/stock-dashboard.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set the following:
     - Repository: `yourusername/stock-dashboard`
     - Branch: `main`
     - Main file path: `app.py`
   - Click "Deploy"

3. **Configuration**
   - Streamlit Cloud will automatically install dependencies from `requirements.txt`
   - Your app will be available at: `https://yourusername-stock-dashboard-app-xyz123.streamlit.app`

## Alternative Deployment Options

### Heroku Deployment

1. **Create additional files:**

   **Procfile**
   ```
   web: sh setup.sh && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

   **setup.sh**
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [general]\n\
   email = \"your-email@domain.com\"\n\
   " > ~/.streamlit/credentials.toml
   echo "\
   [server]\n\
   headless = true\n\
   enableCORS=false\n\
   port = $PORT\n\
   " > ~/.streamlit/config.toml
   ```

2. **Deploy:**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Docker Deployment

1. **Create Dockerfile:**
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8501

   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run:**
   ```bash
   docker build -t stock-dashboard .
   docker run -p 8501:8501 stock-dashboard
   ```

### Railway Deployment

1. **Connect GitHub repository to Railway**
2. **Set environment variables if needed**
3. **Railway will auto-deploy from your repository**

## Environment Variables

If you need to set environment variables for API keys or configuration:

- **Streamlit Cloud**: Use the app settings in Streamlit Cloud
- **Heroku**: Use `heroku config:set VARIABLE_NAME=value`
- **Docker**: Use `-e` flag or environment file

## Performance Optimization

1. **Caching**: The app uses Streamlit's caching for data fetching
2. **Data limits**: Consider limiting historical data range for better performance
3. **Concurrent users**: Streamlit Cloud has usage limits for free tier

## Monitoring and Maintenance

1. **Logs**: Check application logs in your deployment platform
2. **Updates**: Push updates to your GitHub repository for automatic redeployment
3. **Dependencies**: Regularly update `requirements.txt` for security patches

## Troubleshooting

### Common Issues

1. **Module not found errors**
   - Ensure all dependencies are in `requirements.txt`
   - Check Python version compatibility

2. **yfinance connection errors**
   - Add error handling for network issues
   - Consider implementing retry logic

3. **Memory issues**
   - Limit data fetching ranges
   - Implement data pagination

4. **Slow loading**
   - Use Streamlit caching effectively
   - Optimize data processing functions

### Performance Tips

1. **Use st.cache_data for data fetching**
2. **Implement session state for user inputs**
3. **Consider using st.fragments for partial updates**
4. **Optimize chart rendering with plotly**

## Security Considerations

1. **API Keys**: Never commit API keys to repository
2. **Input Validation**: Validate all user inputs
3. **Rate Limiting**: Implement appropriate rate limiting for API calls
4. **HTTPS**: Ensure deployment platform uses HTTPS

## Next Steps

After deployment:
1. **Custom Domain**: Consider using a custom domain
2. **Analytics**: Add usage analytics (Google Analytics, etc.)
3. **User Feedback**: Implement feedback collection
4. **Feature Expansion**: Add more advanced features based on user needs