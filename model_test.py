import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import params as yamnet_params
import yamnet as yamnet_model
# import tensorflow as tf
import json
import requests
import moviepy.editor as mp
import math

# Test Link:    https://www.youtube.com/watch?v=bmeRROzi_4k
MODEL_URI='http://localhost:8501/v1/models/yamnet:predict'


# def segment_find(out_sec_single):
#     PROB_THRESH_CONST = 0.4
#     # ind2 = np.argsort(out_sec[:,CLASS_INDEX])[::-1]
#     print("LOG: SEG: ")
#     print(out_sec_single.shape)
#     ind = np.arange(start = 1, stop = out_sec_single.shape[0])
#     # print(ind.shape)
#     s_time = min(ind,key = lambda x: x if out_sec_single[x] > PROB_THRESH_CONST else 9999)
#     e_time = max(ind,key = lambda x: x if out_sec_single[x] > PROB_THRESH_CONST else -1)
#     #converting into seconds
#     print("Millisecs")
#     print(s_time,e_time)
#     print(out_sec_single[s_time])
#     print("LOG: TESTING: false check")
#     print(out_sec_single[s_time] > PROB_THRESH_CONST)
#     s_time = math.floor(s_time/1000)
#     e_time = math.ceil(e_time/1000)
#     return [s_time,e_time]
def segment_find(out_sec_single):
    PROB_THRESH_CONST = 0.45
    #s_t = duration + 1
    s_t = 9999 
    e_t = 0
    for i in range(out_sec_single.shape[0]):
        if out_sec_single[i] > PROB_THRESH_CONST:
            if s_t > i:
                s_t = i
            if e_t < i:
                e_t = i
    print("Millisecs")
    print(s_t,e_t)  
    print(out_sec_single[s_t])   
    #floor used to get a second earlier before the class detected  
    s_t = math.floor(s_t/1000)
    e_t = math.ceil(e_t/1000)
    return [s_t,e_t]
        

def get_prediction(av_filename,image_path):
    video_flag =0 
    file_ext = av_filename.split('.')[-1]
    if(file_ext == 'mp4' or file_ext == 'mkv' or file_ext == 'webm'):
        video_flag = 1
        # Set a clip value for lesser processing
        CLIP_LENGTH_CONST = 120
        print("LOG: VIDEO READ")
        try:
            clip = mp.VideoFileClip(av_filename)
        except:
            av_filename = av_filename.split('.')[0] + '.mkv'
            clip = mp.VideoFileClip(av_filename)
        #Check if its greater than video length
        if CLIP_LENGTH_CONST > clip.end:
            CLIP_LENGTH_CONST = clip.end
        clip = clip.subclip(0,CLIP_LENGTH_CONST)
        clip.audio.write_audiofile("static/audio.wav",ffmpeg_params = ["-ac","1"])
        # clip.audio.write_audiofile("audio.wav")
        wav_data, sr = sf.read("static/audio.wav", dtype=np.int16)

    elif(file_ext == 'wav'):
        print("LOG: AUDIO READING")
        wav_data, sr = sf.read(av_filename, dtype=np.int16)
    else:
        print("LOG: EXITING")
        exit()
    

    waveform = wav_data / 32768.0
    print(waveform.shape)
    #Find duration
    duration = len(wav_data)/sr
    print(duration)
    #Correction for multi channel audio
    if(waveform.ndim != 1):
        print("LOG: Correction for multi channel audio")
        waveform = waveform[:,1]
        # waveform = waveform.reshape(waveform.shape[0] * waveform.shape[1],)
    # The graph is designed for a sampling rate of 16 kHz, but higher rates should work too.
    # We also generate scores at a 10 Hz frame rate.
    params = yamnet_params.Params(sample_rate=sr, patch_hop_seconds=0.1)
    print("Sample rate =", params.sample_rate)

    # Set up the YAMNet model.
    class_names = yamnet_model.class_names('yamnet_class_map.csv')


    # Run the model.
    data = json.dumps(
        {"inputs" : waveform.tolist()}
    )
    response = requests.post(MODEL_URI,data)
    out = response.json()

    out_numpy = np.array(out['outputs']['activation_2'])

    # print(out['error'])
    print("LOG: output Shape")
    print(out_numpy.shape)

    #Converting into milliseconds 
    duration = int(duration)
    out_sec = np.zeros(shape=(duration * 1000,out_numpy.shape[1]))
    # out_sec = np.zeros(shape=(duration,out_numpy.shape[1]))
    for i in range(0,out_sec.shape[0]):
        index = ((i)/out_sec.shape[0]) * out_numpy.shape[0]
        index = int(index)
        # print(index,i)
        out_sec[i] = out_numpy[index]

    
    #  Plot and label the model output out_numpy for the top-scoring classes.
    plt.figure(figsize=(10, 8))
    mean_out_numpy = np.mean(out_numpy, axis=0)
    top_N = 10
    top_class_indices = np.argsort(mean_out_numpy)[::-1][:top_N]
    plt.subplot(3, 1, 3)
    save_img = plt.gcf()
    plt.imshow(out_sec[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
    # Compensate for the patch_window_seconds (0.96s) context window to align with spectrogram.
    patch_padding = (params.patch_window_seconds / 2) / params.patch_hop_seconds
    plt.xlim([-patch_padding, out_sec.shape[0] + patch_padding])
    # Label the top_N classes.
    yticks = range(0, top_N, 1)
    plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
    _ = plt.ylim(-0.5 + np.array([top_N, 0]))
    save_img.savefig(image_path,bbox_inches = 'tight')

    # explotion : 420 to 425
    # scream : 11
    # Siren : 390
    CLASS_INDEX = 420
    if CLASS_INDEX in top_class_indices:
        li = segment_find( out_sec[:,CLASS_INDEX] ) 
        print("LOG:#### (s_time,e_time)")
        print(li)
        #Cropping and writing onto a file 
        if video_flag == 1:
            clip = clip.subclip(li[0],li[1])
            clip.write_videofile(filename= "static/out_vid_clipped.mp4")
            clip.close()
        else:
            clip = mp.AudioFileClip(av_filename).subclip(li[0],li[1])
            clip.write_audiofile(filename = "static/out_audio_clipped.wav")
            clip.close()

    out = class_names[top_class_indices[0]] + ' ' + class_names[top_class_indices[1]] + ' and clipped ' + str(li[0]) + ' to ' + str(li[1]) + 'seconds'
    return out



