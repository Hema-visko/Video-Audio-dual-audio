import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence
# import asyncio
import whisper
import warnings
import wave
import audioop
import io
import base64
import json
import matplotlib.pyplot as plt
import math
import librosa.display
from flask import Flask, request, jsonify
import cv2
import tempfile
import os
from ultralytics import YOLO
import json
from deepface import DeepFace
import shutil
from werkzeug.utils import secure_filename
import random


# import psutil
from flask import Flask, jsonify, request,send_file
from io import BytesIO
import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import cv2
import pandas as pd
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import signal
import sys


numbers = random.randint(1,10)

# Monkey patch SIGKILL for Windows compatibility with NeMo
if sys.platform == "win32" and not hasattr(signal, "SIGKILL"):
    signal.SIGKILL = signal.SIGTERM  # Fallback to something Windows supports



from nemo.collections.asr.models import SortformerEncLabelModel
from pydub import AudioSegment

app = Flask(__name__)


google_api_key =  'AIzaSyCaU7bKNQGTt-FyNkW6nIAUM-p_sbjgpUw'

# google_api_key = 'AIzaSyAztskJ1eyTMCzsHJVje4o9XZAybI4F488'

llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', temperature=0.6, google_api_key=google_api_key)

memory = ConversationBufferMemory()
conversation_chain = ConversationChain(llm=llm, memory=memory)

whisper_model = whisper.load_model("small")


diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")


diar_model = SortformerEncLabelModel.restore_from(restore_path="C:/Users/pc/OneDrive/Desktop/Video_object_detection/diar_sortformer_4spk-v1.nemo")



diar_model.eval()



def convert_mp4_to_wav(input_file, output_file):
   
    audio = AudioSegment.from_file(input_file, format="mp4")
    audio.export(output_file, format="wav")

def video_analyzer(video_path):
  try:
      #  Trained model
      model_best = load_model('/home/ritik/audio/face_model.h5')  # Set your model file path

      # Classes for 7 emotional states
      class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

      # Load the pre-trained face cascade
      face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


      output_folder = 'tcs_'  # Folder to save frames
      os.makedirs(output_folder, exist_ok=True)

      # Clear the output folder before saving new frames
      for file in os.listdir(output_folder):
          file_path = os.path.join(output_folder, file)
          if os.path.isfile(file_path):
              os.remove(file_path)


      csv_path = 'emotion_results_tcs.csv'


      cap = cv2.VideoCapture(video_path)
      total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

      target_frames = sorted(random.sample(range(total_frames), min(255, total_frames)))
      frame_indices = set(target_frames)
      frame_count = 0

      # Dictionary to store results
      results = {}

      while cap.isOpened():
          ret, frame = cap.read()
          if not ret:
              break

          if frame_count in frame_indices:
              # Save frame
              frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
              cv2.imwrite(frame_filename, frame)

              # Convert the frame to grayscale for face detection
              gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

              # Detect faces in the frame
              faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

              # Process each detected face
              for (x, y, w, h) in faces:
                  face_roi = frame[y:y + h, x:x + w]
                  face_image = cv2.resize(face_roi, (48, 48))
                  face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                  face_image = image.img_to_array(face_image)
                  face_image = np.expand_dims(face_image, axis=0)
                  face_image = np.vstack([face_image])

                  # Predict emotion
                  predictions = model_best.predict(face_image)
                  emotion_label = class_names[np.argmax(predictions)]

                  # Overwrite result
                  results[frame_filename] = emotion_label

          frame_count += 1

  

      results_df = pd.DataFrame(list(results.items()), columns=['Frame', 'Emotion'])
      results_df.to_csv(csv_path, index=False)

      # Count occurrences of each emotion
      face_emotion = results_df['Emotion'].value_counts().to_dict()
      print(face_emotion)


      positive_emotions = ['Happy', 'Surprise', 'Neutral']
      negative_emotions = ['Sad', 'Angry', 'Fear']
      positive = []
      negative = []
      total_emotions = []
      positive_detected = []
      negative_detected = []


      for emotions,num in face_emotion.items():
          total_emotions.append(num)
          if emotions in positive_emotions:
              positive_detected.append(num)
          elif emotions in negative_emotions:
              negative_detected.append(num)
          

      total_emotions_count = sum(total_emotions)
      print("Positive emotions detected:", positive_detected)
      print("Negative emotions detected:", negative_detected)

      print(total_emotions_count)

      pos=  (sum(positive_detected)/total_emotions_count)*100
      neg = (sum(negative_detected)/total_emotions_count)*100
      overall_score = pos * (1-(neg/100))

      print("no emotion detected")
      print('positive score',positive_detected)
      print('negative score',negative_detected)
      data = {
          "overall_video_score": overall_score,
          "positive_emotions_score": pos,
          "negative_emotions_score": neg,
      }
      
      return data

  except Exception as e:
        print(f"Error occurred: {e}")
        return 'No emotion detected'


def load_audio(file):
    audio, sr = librosa.load(file, sr=None)
    return audio, sr


# Convert NumPy array to AudioSegment
def numpy_to_audiosegment(audio, sr):
    audio = (audio * (2**15 - 1)).astype(np.int16)  # Convert to 16-bit PCM format
    audio_segment = AudioSegment(
        audio.tobytes(),
        frame_rate=sr,
        sample_width=audio.dtype.itemsize,
        channels=1
    )
    return audio_segment


# Analyze pitch and tone
def analyze_pitch_and_tone(audio, sr):
    pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0])  # Ignore zero values
    pitch_std = np.std(pitches[pitches > 0])
    return pitch_mean, pitch_std


# Analyze speech rate
def analyze_speech_rate(transcript, audio_duration):
    words = transcript.split()
    word_count = len(words)
    speech_rate = word_count / audio_duration
    return speech_rate


# Detect pauses and fillers
def detect_pauses_and_fillers(audio_segment, transcript):
    # Detect silence (pauses)
    silence_threshold = -40  # dB
    min_silence_len = 500  # ms
    pauses = detect_silence(audio_segment, min_silence_len, silence_threshold, 1)

    # Count fillers ("um", "uh")
    fillers = ["um", "uh"]
    filler_count = sum(transcript.lower().count(filler) for filler in fillers)

    return pauses, filler_count


# Analyze volume and clarity
def analyze_volume_and_clarity(audio):
    rms = librosa.feature.rms(y=audio)[0]
    volume_mean = np.mean(rms)
    volume_std = np.std(rms)
    return volume_mean, volume_std


# Audio to text using Whisper
def transcribe_audio_whisper(file):
        
    try:
        warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
        result = whisper_model.transcribe(file, fp16=False)
        return result["text"]
    except:
        return "No transcript is avaliable"


# Graph function is here
def analyze_pitch(audio_file, sr=22050):
    """
    Analyze the pitch (fundamental frequency) of an audio file.
    """
    y, sr = librosa.load(audio_file, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    pitch = []
    times = []

    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch_val = pitches[index, t]
        if pitch_val > 0:  # Ignore unvoiced regions (pitch=0)
            pitch.append(pitch_val)
            times.append(t * 512 / sr)

    return np.array(times), np.array(pitch)

def plot_pitch(times, pitch, duration=10, title="Pitch Analysis"):
    """
    Plot the pitch over a specified duration and save the plot to a BytesIO object.
    """
    mask = times <= duration
    times_filtered = times[mask]
    pitch_filtered = pitch[mask]

    max_value = max(pitch_filtered)
    mean_value = np.mean(pitch_filtered)

    # Plot the filtered data
    plt.figure(figsize=(12, 6))
    plt.plot(times_filtered, pitch_filtered, label="Pitch", color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"{title} (First {duration} seconds)")
    plt.legend()
    plt.grid(True)

    # Annotate time and pitch values every 2 seconds
    for t in range(0, int(duration) + 1, 2):
        idx = (np.abs(times_filtered - t)).argmin()
        plt.annotate(f"{pitch_filtered[idx]:.1f} Hz",
                     (times_filtered[idx], pitch_filtered[idx]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=8,
                     color="green")

    # Add max and mean value annotations
    plt.axhline(max_value, color="green", linestyle="--", label=f"Max: {max_value:.1f} Hz")
    plt.axhline(mean_value, color="orange", linestyle="--", label=f"Mean: {mean_value:.1f} Hz")
    plt.legend()

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return buffer


def analyze_audio(file_path):

    audio, sr = load_audio(file_path)
    audio_duration = librosa.get_duration(y=audio, sr=sr)

    # Convert audio to AudioSegment
    audio_segment = numpy_to_audiosegment(audio, sr)

    # Transcribe audio with Whisper
    transcript = transcribe_audio_whisper(file_path)
    if not transcript:
        # print("No transcript available for analysis.")
        return

    # Analyze features
    pitch_mean, pitch_std = analyze_pitch_and_tone(audio, sr)
    speech_rate = analyze_speech_rate(transcript, audio_duration)
    pauses, filler_count = detect_pauses_and_fillers(audio_segment, transcript)
    volume_mean, volume_std = analyze_volume_and_clarity(audio)

    # Analyze basic properties of the new audio file
    new_audio_properties = {}
    try:
        with wave.open(file_path, 'r') as audio_:
            # Extract basic audio properties
            new_audio_properties['frame_rate'] = audio_.getframerate()
            new_audio_properties['n_frames'] = audio_.getnframes()
            new_audio_properties['time_duration(sec)'] = audio_.getnframes() / audio_.getframerate()

            # Compute average loudness (RMS) and silence ratio
            frames = audio_.readframes(audio_.getnframes())
            rms = audioop.rms(frames, audio_.getsampwidth())  # Root mean square of the audio signal
            silent_frames = sum(
                audioop.rms(frames[i:i + audio_.getsampwidth()], audio_.getsampwidth()) < 1000
                for i in range(0, len(frames), audio_.getsampwidth())
            )
            silence_ratio = silent_frames / len(frames)
    except Exception as e:
        new_audio_properties['error'] = str(e)

    # Package detailed results for the new audio file
    new_audio_properties.update({
        "average_voice_loudness": rms,
        "silence_ratio": silence_ratio,
        "speech_detected": silence_ratio < 0.5,  # Assumes speech if less than 50% silence
    })

    # Output the analysis results
    # print(new_audio_properties)

    scores = []

    if 10 <= len(pauses) <= 20:
        scores.append('Average')
    elif 1 <= len(pauses) <= 10:
        scores.append('Good')
    elif len(pauses) == 0:
        scores.append('Excellent')
    elif len(pauses) > 20:
        scores.append('Bad')

  
    if speech_rate <= 1.25:
        scores.append('Bad')
    elif 1.25 <= speech_rate <= 2.05:
        scores.append('Average')
    elif 2.05 <= speech_rate <= 3.0:
        scores.append('Good')
    else:
        scores.append('Excellent')

    if silence_ratio > 0.45:
        scores.append('Bad')
    elif 0 <= silence_ratio <= 0.20:
        scores.append('Excellent')
    elif 0.20 < silence_ratio <= 0.35:
        scores.append('Good')
    elif 0.35 < silence_ratio <= 0.45:
        scores.append('Average')


    if pitch_mean <= 732:
        scores.append('Bad')
    elif 732 < pitch_mean <= 900:
        scores.append('Average')
    elif pitch_mean > 900:
        scores.append('Good')

    if pitch_std <= 723:
        scores.append('Bad')
    elif 723 < pitch_std <= 900:
        scores.append('Average')
    elif pitch_std > 900:
        scores.append('Good')


    if pitch_mean > 1350 or rms > 2800:
        pitch_mean = pitch_mean / 1.25
        rms = rms / 1.75

        Average_positive = [pitch_mean, pitch_std, speech_rate, rms]
        Average_negative = [len(pauses), (silence_ratio * 10)]

        Scaled_positive = sum(Average_positive) / 4
        Scaled_negative = sum(Average_negative) / 2

        Scaled_positive_1000 = Scaled_positive / 1000
        Scaled_negative_100 = Scaled_negative / 100

        final_conversion_p = Scaled_positive_1000 * 0.25
        final_conversion_n = Scaled_negative_100 * 0.50

        Result = final_conversion_p - final_conversion_n
        # print(final_conversion_p, '\n', final_conversion_n)

        Result_new = Result * 2.5
        final_result = Result_new * 100
        # print(f'Your final Audio Score is {final_result}')

    else:
        Average_positive = [pitch_mean, pitch_std, speech_rate]
        Average_negative = [len(pauses), (silence_ratio * 10)]

        Scaled_positive = sum(Average_positive) / 3
        Scaled_negative = sum(Average_negative) / 2

        Scaled_positive_1000 = Scaled_positive / 1000
        Scaled_negative_100 = Scaled_negative / 100

        final_conversion_p = Scaled_positive_1000 * 0.25
        final_conversion_n = Scaled_negative_100 * 0.50

        Result = final_conversion_p - final_conversion_n
        Result_new = Result * 2.5
        final_result = Result_new * 100



    audio_score = []

    if scores.count('Bad') >= 2:
        final_result_new = final_result / 2
        audio_score.append(final_result_new)

    elif scores.count('Bad') == 1:
        decrease = final_result * (1 / 4)
        final_result_new = final_result - decrease
        audio_score.append(final_result_new)
        # print(f'Your final Audio Score is {final_result_new}')
    else:
        # print(f'Your final Audio Score is {final_result}')
        audio_score.append(final_result)

      # Define the prompt for analysis
    analysis_prompt = f""" {transcript} analysis the text based on grammar mistakes, fluency, clarity , professional tone and  overall rate it and rate out of 100 this is the text of interviewer analysis it completely Provide me only the scores don't give unnecessary text also detect irrelevant words means(out of context words or out of topic like here is candidate self introduction text if find irrelevant topic than self introduction) and give me score if there are out of context words and drop overall score if there is irrelvancy return only overall score not return any other score give me final score by decreasing the (give me irrelevancy score in positive number only don't give negative number return in json format)
    """
    # print(transcript)

    def chat_with_bot(user_input):
        formatted_prompt = analysis_prompt.format(text=user_input)
        response = conversation_chain.run(input=formatted_prompt)
        return response

    response = chat_with_bot(user_input=analysis_prompt)

    start = response.replace("```json", "")
    end = start.replace("```", "")
    end1 = json.loads(end)

    text_final_score = end1["overall"] - end1["irrelevancy"]
    audio_final_score = audio_score[0]

    overall_final_score = (text_final_score + audio_final_score) / 2
    overall_final_score = int(overall_final_score)


    audio_detail={
       "Pitch Mean:": f"{pitch_mean:.2f}",
        "Pitch Std:" :f"{pitch_std:.2f}",
        "Speech Rate:": f"{speech_rate:.2f}",
        "Number of Pauses:": f"{len(pauses)}",
        "Filler Count words (um,uh):" :f"{filler_count}",
        "Audio properties:":f"{new_audio_properties}"

    }

    # print(audio_detail)
    if overall_final_score<0:
        return (numbers, json.dumps(audio_detail, indent=4))
    else:
        return (overall_final_score,json.dumps(audio_detail, indent=4))



def diarize_audio(wav_path):
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"File not found: {wav_path}")

    audio = AudioSegment.from_wav(wav_path)
    audio = audio.set_channels(1).set_frame_rate(16000)


    processed_audio_path = wav_path.replace(".wav", "_processed.wav")
    audio.export(processed_audio_path, format="wav")
        

    predicted_segments = diar_model.diarize(
        audio=processed_audio_path,
        batch_size=1,
        include_tensor_outputs=True
    )

    return predicted_segments

def deep_tensor_to_python(obj):
    if hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: deep_tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_tensor_to_python(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(deep_tensor_to_python(i) for i in obj)
    return obj





model = YOLO("yolo11n.pt")  # Update path as needed
TARGET_CLASSES = ["person", "laptop", "cell phone", 'remote','book']

# @app.route('/analyze_video', methods=['POST'])
def analyze_video(video_path):
    # Check if video file was uploaded
    # if 'video' not in request.files:
    #     return jsonify({"error": "No video file provided"}), 400
    
    video_file = video_path
    
    # Validate file extension
    # if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
    #     return jsonify({"error": "Invalid file format. Please upload a video file"}), 400

    # Save the uploaded video to a temporary file
    # temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    # video_file.save(temp_video.name)
    # # temp_video.close()

    # Initialize counters
    total_person_count = 0
    objects_found = {
        "laptop": 0,
        "cell phone": 0,
        "remote": 0,
        "book": 0
    }   
    frame_count = 0

    # Process the video
    cap = cv2.VideoCapture(video_file)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0].boxes.data.cpu().numpy()

        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            class_id = int(class_id)
            class_name = model.names[class_id]

            if class_name in TARGET_CLASSES:
                if class_name == "person" and conf > 0.6:
                    total_person_count += 1
                elif class_name in objects_found and conf > 0.45:
                    objects_found[class_name] += 1

        frame_count += 1

    cap.release()
    # os.unlink(temp_video.name)  # Delete the temporary file

    # Calculate averages
    avg_person_per_frame = total_person_count / frame_count if frame_count > 0 else 0

    response = {
        "status": "success",
        "analysis": {
            "total_persons_detected": total_person_count,
            "average_persons_per_frame": round(avg_person_per_frame, 2),
            "objects_detected": objects_found,
            "total_frames_processed": frame_count
        }
    }

    def video_to_frames(video_path, output_folder):
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Load the video
        cap = cv2.VideoCapture(video_path)
        count = 0

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        print("Extracting frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame as image
            frame_filename = os.path.join(output_folder, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            count += 1

        cap.release()
        print(f"Done! {count} frames saved in '{output_folder}'.")

    folder_name = 'output_frames'

    # Example usage
    video_to_frames(video_file, folder_name)



    '''Face Recognitions'''
    frames_path = folder_name
    frames = os.listdir(frames_path)

    # Ensure the frames are sorted if needed
    frames.sort()
    same_person_count = 0
    diff_person_count = 0
    no_person_count = 0

    try:
        if len(frames) < 2:
            print("Not enough frames to compare.")
        else:
            # We'll use the first frame as the reference frame
            reference_frame = frames[0]
            reference_path = os.path.join(frames_path, reference_frame)

            for i in range(1, len(frames)):
                try:
                    current_path = os.path.join(frames_path, frames[i])
                    result = DeepFace.verify(
                        img1_path=reference_path,
                        img2_path=current_path
                    )
                    print(f"Comparing {reference_frame} with {frames[i]}:")
                    if result.get('verified'):
                        same_person_count += 1
                        print("✅ Same person")
                    else:
                        diff_person_count += 1
                        print("❌ Different persons")

                except Exception as e:
                    print(f"Error comparing {frames[i]}: {e}")
                    no_person_count += 1
    except Exception as e:
        print(f"General error: {e}")

    verified_result = {
        'Same persons count': same_person_count,
        'Different persons count': diff_person_count,
        'No person detected count': no_person_count
    }


    # Print as JSON-formatted string
    # print(json.dumps(verified_result, indent=4))
    # face_detection = json.dumps(verified_result,indent=4)


    shutil.rmtree(frames_path)
    return (response,verified_result)







@app.route('/analyzer', methods=['POST'])
def analyze_audio_video():
    try: 
        # File validation
        if 'file' not in request.files:
            return jsonify({"message": "No file provided", "status": False}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"message": "No file selected", "status": False}), 400
        
        if not file.filename.lower().endswith(".mp4"):
            return jsonify({"message": "Only MP4 files are supported", "status": False}), 400

        # Create upload directory
        upload_folder = "./myaud"
        os.makedirs(upload_folder, exist_ok=True)
        
        # Save file securely
        video_path = os.path.join(upload_folder, secure_filename(file.filename))
        file.save(video_path)

        

        data = cv2.VideoCapture(video_path) 
            

        frames = data.get(cv2.CAP_PROP_FRAME_COUNT) 
        fps = data.get(cv2.CAP_PROP_FPS) 

        
        # calculate duration of the video 
        seconds = round(frames / fps) 
        if seconds<=80:
            video_time = datetime.timedelta(seconds=seconds) 
            print(f"duration in seconds: {seconds}") 
            print(f"video time: {video_time}") 
        else:
            return 'Video Duration is Higher it must be less that 1.10 minutes'

        
        audio_path = os.path.join(upload_folder, "audio_output.wav")
        convert_mp4_to_wav(video_path, audio_path)

        # Speaker diarization
        segments, raw_probs = diarize_audio(audio_path)
        speaker_analysis = deep_tensor_to_python(segments)

        # Parse speaker segments
        unique_speakers = set()
        if isinstance(speaker_analysis, list):
            for segment_group in speaker_analysis:
                if isinstance(segment_group, list):
                    for segment in segment_group:
                        if isinstance(segment, str) and "speaker_" in segment:
                            speaker = segment.split()[-1]
                            unique_speakers.add(speaker)

        # Immediate termination for multiple speakers
        if len(unique_speakers) > 1:
            return jsonify({
                "message": "Multiple speakers detected - analysis terminated",
                "detected_speakers": list(unique_speakers),
                "speaker_analysis": speaker_analysis,
                "status": False
            }), 400

        # Only proceed with full analysis if exactly one speaker
        times, pitch = analyze_pitch(audio_path)
        buffer = plot_pitch(times, pitch, duration=10, title="Pitch Analysis")
        buffer.seek(0)

        audio_score, audio_detail = analyze_audio(audio_path)
        video_score = video_analyzer(video_path)

        # Convert to Python-native types
        audio_score = deep_tensor_to_python(audio_score)
        video_score = deep_tensor_to_python(video_score)
        audio_detail = deep_tensor_to_python(audio_detail)

        # Format audio_detail into desired structure
        if isinstance(audio_detail, str):
            audio_detail = json.loads(audio_detail)

        audio_detail_formatted = {
            "Pitch Mean:": str(audio_detail.get("Pitch Mean:", "")),
            "Pitch Std:": str(audio_detail.get("Pitch Std:", "")),
            "Speech Rate:": str(audio_detail.get("Speech Rate:", "")),
            "Number of Pauses:": str(audio_detail.get("Number of Pauses:", "")),
            "Filler Count words (um,uh):": str(audio_detail.get("Filler Count words (um,uh):", "")),
            "Audio properties:": str(audio_detail.get("Audio properties:", ""))
        }

        # Score normalization helper
        def normalize_score(score):
            if isinstance(score, (int, float)):
                return float(score)
            elif isinstance(score, dict):
                return float(score.get('overall_score', score.get('score', 0)))
            elif isinstance(score, (list, tuple)) and len(score) > 0:
                return float(sum(score) / len(score))
            return 0.0

        audio_val = normalize_score(audio_score)
        video_val = normalize_score(video_score)
        final_score = round((audio_val + video_val) / 2, 2)

        object_detection = analyze_video(video_path)

        # Successful response
        return jsonify({
            "audio": audio_val,
            "audio_detail": audio_detail_formatted,
            "video_analysis": video_score,
            "speaker_analysis": speaker_analysis,
            "object_detection": object_detection,
            "final_overall_score": final_score,
            "message": "Analysis completed successfully",
            "status": True
        }), 200

    except Exception as e:
        print(f"Error: {e}", flush=True)
        return jsonify({
            "message": f"Processing error: {str(e)}",
            "status": False
        }), 500
    
@app.route('/graph', methods=['POST'])
def graph_analysis():
    try: 
        file = request.files.get("file")
        if not file:
            return jsonify({
                "message": "No file provided",
                "status": False
            }), 400
        

        if not file.filename.endswith(".mp4"):
            return jsonify({
                "message": "This is not an MP4 file",
                "status": False
            }), 400
        
        # Save the video file
        upload_folder = "./myaud"

        os.makedirs(upload_folder, exist_ok=True)

        video_path = os.path.join(upload_folder, file.filename)


        # Save the video file
        # video_path = f"./upload/{file.filename}"
        file.save(video_path)
        
        # Define the audio path
        audio_path = f"./myaud/audio_output.wav"
 
        convert_mp4_to_wav(video_path, audio_path)


        times, pitch = analyze_pitch(audio_path)

        # # Generate pitch plot
        buffer = plot_pitch(times, pitch, duration=10, title="Pitch Analysis")

        # Clean up the temporary file
        # os.remove(imagepath)
        buffer.seek(0)

        return send_file(buffer, mimetype='image/png')


    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "message": "An error occurred during processing",
            "status": False
        }), 500


if __name__ == "__main__":
    app.run(debug=True,port=8000)
