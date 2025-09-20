# COIN Video Download Script

Fixed version of the COIN video download script that addresses YouTube authentication issues.

## 🐛 Problems Fixed

1. **YouTube OAuth Authentication**: Removed deprecated OAuth2 authentication
2. **Format Selection**: Changed from `-f mp4` to `best[ext=mp4]/best` for better compatibility
3. **Error Handling**: Added robust error handling and retry mechanisms
4. **Rate Limiting**: Reduced default workers from 16 to 4 to avoid YouTube rate limits
5. **Cookie Authentication**: Added support for browser cookies and cookie files

## 🚀 Usage

### Method 1: Using Browser Cookies (Recommended)
```bash
# Uses Chrome cookies automatically
python download_videos.py --output_dir datasets/coin/videos --json_path datasets/coin/coin.json

# Use different browser
python download_videos.py --browser firefox
```

### Method 2: Using Cookie File
```bash
# First, export cookies
python export_cookies.py --browser chrome --output youtube_cookies.txt

# Then use the cookies file
python download_videos.py --cookies youtube_cookies.txt
```

### Method 3: Custom Configuration
```bash
python download_videos.py \
    --output_dir datasets/coin/videos \
    --json_path datasets/coin/coin.json \
    --num_workers 2 \
    --cookies youtube_cookies.txt \
    --ffmpeg /path/to/ffmpeg
```

## 📋 Parameters

- `--output_dir`: Directory to save downloaded videos (default: `datasets/coin/videos`)
- `--json_path`: Path to COIN JSON file (default: `datasets/coin/coin.json`)
- `--num_workers`: Number of parallel downloads (default: 4, reduced from 16)
- `--cookies`: Path to cookies file for YouTube authentication
- `--browser`: Browser to extract cookies from (chrome, firefox, safari, edge)
- `--ffmpeg`: Path to ffmpeg executable

## 🔧 Troubleshooting

### YouTube Authentication Issues
1. **Login to YouTube** in your browser first
2. **Export cookies** using the helper script:
   ```bash
   python export_cookies.py --browser chrome
   ```
3. **Use the cookies file**:
   ```bash
   python download_videos.py --cookies youtube_cookies.txt
   ```

### Rate Limiting
- Reduce `--num_workers` to 1-2 if getting rate limited
- Add delays between requests if needed

### Format Issues
- The script now uses `best[ext=mp4]/best` format selection
- This prefers MP4 but falls back to best available format
- No more format warnings!

### Network Issues
- Check your internet connection
- Some videos may be region-restricted
- The script will retry failed downloads without cookies

## 📊 Improvements Made

1. **Better Error Handling**: Individual video failures don't stop the entire process
2. **Retry Mechanism**: Failed downloads are retried without cookies
3. **Progress Logging**: Clear status messages for each download
4. **Format Optimization**: Uses recommended format selection
5. **Rate Limiting**: Reduced concurrent downloads to avoid blocking
6. **Cookie Support**: Multiple authentication methods

## 🎯 Expected Behavior

- Videos download to `datasets/coin/videos/` by default
- Existing videos are verified and re-downloaded if corrupted
- Failed downloads are logged but don't stop the process
- Progress is shown for each successful download
- The script handles YouTube's authentication requirements gracefully