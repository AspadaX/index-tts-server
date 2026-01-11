import requests
import time
import os

BASE_URL = "http://localhost:8000/api/v1"

def test_register():
    print("Testing registration...")
    voice_path = "index-tts/examples/voice_01.wav"
    if not os.path.exists(voice_path):
        print(f"Voice file not found: {voice_path}")
        return None
        
    with open(voice_path, "rb") as f:
        files = {"file": f}
        data = {"speaker_name": "TestSpeaker"}
        resp = requests.post(f"{BASE_URL}/register_speaker_audio", files=files, data=data)
        print(resp.json())
        return resp.json().get("status") == "Success"

def test_list():
    print("Testing list speakers...")
    resp = requests.get(f"{BASE_URL}/get_speakers")
    print(resp.json())
    speakers = resp.json().get("data", {}).get("speakers", [])
    if speakers:
        return speakers[0][1] # Return uuid of first speaker
    return None

def test_generate(speaker_uuid):
    print("Testing generate...")
    data = {
        "speaker_uuid": speaker_uuid,
        "text": "Hello world, this is a test."
    }
    resp = requests.post(f"{BASE_URL}/generate", json=data)
    print(resp.json())
    return resp.json().get("data", {}).get("task_id")

def test_status(task_id):
    print(f"Testing status for {task_id}...")
    while True:
        resp = requests.get(f"{BASE_URL}/generate", params={"task_id": task_id})
        
        content_type = resp.headers.get("content-type", "")
        if "audio" in content_type:
            print("Task completed, audio received.")
            with open("test_output.wav", "wb") as f:
                f.write(resp.content)
            return True
        
        try:
            json_resp = resp.json()
            status = json_resp.get("status")
            print(f"Status: {status}")
            
            if status == "Failed":
                print(f"Task failed: {json_resp.get('message')}")
                return False
                
            if status == "Success":
                # Should have returned file
                print("Status success but JSON returned?")
                pass
        except:
            print(f"Error parsing response: {resp.text}")
            
        time.sleep(2)

if __name__ == "__main__":
    try:
        if test_register():
            speaker_uuid = test_list()
            if speaker_uuid:
                task_id = test_generate(speaker_uuid)
                if task_id:
                    test_status(task_id)
    except Exception as e:
        print(f"Test failed: {e}")
