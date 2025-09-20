import json, os, argparse, subprocess, random, torchvision
import concurrent.futures
try:
    torchvision.set_video_backend('video_reader')
except:
    import av # otherwise, check if av is installed

def download_video(video_id, video_url, output_dir, ffmpeg_location=None, cookies_file=None):
    output_path = os.path.join(output_dir, f'{video_id}.mp4')
    if os.path.exists(output_path):
        try:
            ffmpeg_cmd = ["ffmpeg", "-v", "error", "-i", output_path, "-f", "null", "-"]
            if ffmpeg_location:
                ffmpeg_cmd[0] = os.path.join(ffmpeg_location, "ffmpeg")
            subprocess.run(ffmpeg_cmd, check=True)
            print(f'{output_path} has been downloaded and verified...')
            return
        except:
            print(f'{output_path} may be broken. Downloading it again...')
            os.remove(output_path)
    
    # Build yt-dlp command with improved options
    cmd = [
        "yt-dlp",
        "--no-warnings",  # Suppress warnings
        "--format", "best[ext=mp4]/best",  # Prefer mp4, fallback to best available
        "--output", output_path,
        "--no-playlist",  # Don't download playlists
        "--ignore-errors",  # Continue on individual video errors
    ]
    
    # Add cookies if provided
    if cookies_file and os.path.exists(cookies_file):
        cmd.extend(["--cookies", cookies_file])
    else:
        # Try to use browser cookies as fallback
        cmd.extend(["--cookies-from-browser", "chrome"])  # Try Chrome first
    
    # Add ffmpeg location if specified
    if ffmpeg_location:
        cmd.extend(["--ffmpeg-location", ffmpeg_location])
    
    # Add the video URL
    cmd.append(video_url)
    
    try:
        # Run yt-dlp and capture both stdout and stderr
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f'✅ Successfully downloaded: {video_id}')
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else "No error message"
        stdout_msg = e.stdout.strip() if e.stdout else ""
        
        print(f'❌ Failed to download {video_id}')
        if error_msg:
            print(f'   Error: {error_msg}')
        if stdout_msg:
            print(f'   Output: {stdout_msg}')
        
        # Try alternative approach without cookies
        if cookies_file:
            print(f'🔄 Retrying {video_id} without cookies...')
            cmd_no_cookies = [c for c in cmd if not c.startswith('--cookies')]
            try:
                subprocess.run(cmd_no_cookies, check=True, capture_output=True, text=True)
                print(f'✅ Successfully downloaded {video_id} without cookies')
            except subprocess.CalledProcessError as e2:
                error_msg2 = e2.stderr.strip() if e2.stderr else "No error message"
                print(f'❌ Final failure for {video_id}: {error_msg2}')
        else:
            # Try with different format
            print(f'🔄 Retrying {video_id} with different format...')
            cmd_alt = [c for c in cmd if c != "--format" and c != "best[ext=mp4]/best"]
            cmd_alt.extend(["--format", "best"])
            try:
                subprocess.run(cmd_alt, check=True, capture_output=True, text=True)
                print(f'✅ Successfully downloaded {video_id} with alternative format')
            except subprocess.CalledProcessError as e3:
                error_msg3 = e3.stderr.strip() if e3.stderr else "No error message"
                print(f'❌ Final failure for {video_id}: {error_msg3}')
    except Exception as e:
        print(f'❌ Unexpected error downloading {video_id}: {e}')

def main(output_dir, json_path, num_workers, ffmpeg_location, cookies_file=None):
    # Check if JSON file exists
    if not os.path.exists(json_path):
        print(f"❌ Error: JSON file not found: {json_path}")
        print(f"💡 Please provide a valid path to the COIN dataset JSON file")
        print(f"📝 Example: python download_videos.py --json_path sample_coin.json")
        return
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'database' not in data:
            print(f"❌ Error: Invalid JSON format. Expected 'database' key in {json_path}")
            return
            
        annotations = data['database']
        annotations = list(annotations.items())
        random.shuffle(annotations)
        
        print(f"✅ Loaded {len(annotations)} videos from {json_path}")
        print(f"📁 Output directory: {output_dir}")
        if cookies_file:
            print(f"🍪 Using cookies file: {cookies_file}")
        else:
            print("🌐 Using browser cookies (Chrome)")
        
        print(f"🚀 Starting download with {num_workers} workers...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(download_video, video_id, annotation['video_url'], output_dir, ffmpeg_location, cookies_file) for video_id, annotation in annotations]
            concurrent.futures.wait(futures)
        
        print("✅ Download process completed!")
        
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format in {json_path}: {e}")
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download videos in parallel using yt-dlp')
    parser.add_argument('--output_dir', type=str, default='datasets/coin/videos', help='Directory to save downloaded videos')
    parser.add_argument('--json_path', type=str, default='datasets/coin/coin.json', help='Path to the JSON file containing video data')
    parser.add_argument('--ffmpeg', type=str, default=None, help='Path to ffmpeg executable')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel downloads (reduced from 16 to avoid rate limiting)')
    parser.add_argument('--cookies', type=str, default=None, help='Path to cookies file for YouTube authentication')
    parser.add_argument('--browser', type=str, default='chrome', choices=['chrome', 'firefox', 'safari', 'edge'], help='Browser to extract cookies from')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    main(args.output_dir, args.json_path, args.num_workers, args.ffmpeg, args.cookies)