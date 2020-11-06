import youtube_dl
import moviepy.editor as mp

  
def dwl_vid(url,clip_length = 120): 
    print(url)
    ydl_opts ={
        'format': 'bestvideo[height<=480]+bestaudio/best',
        'outtmpl' : 'static/inp_vid.%(ext)s',
            }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url,download=True) 
            #Video Metadata into result
            # ydl.download(['https://www.youtube.com/watch?v=MZWina8plDk'])

    # Must b same as outtmpl format
    # video_filename = 'input_video.' + result['ext']
    print("LOG:###### Extension: "+result['ext'])
    video_filename = 'static/' + 'inp_vid.' + result['ext']
    print(video_filename)
    return video_filename, result['title']