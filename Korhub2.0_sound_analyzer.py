import streamlit as st
import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from audio_recorder_streamlit import audio_recorder

st.title("Korhub 음성분석기 입니다.")

service = ['종합분석', '모음분석', '자음분석', '문장분석']

selected_service = st.selectbox('원하시는 서비스를 선택해주세요.', service)

if selected_service == '자유음성분석':
    st.write('자유음성분석 서비스를 선택하셨습니다. 기본정보를 기입해주세요.')
elif selected_service == '모음분석':
    st.write('모음분석 서비스를 선택하셨습니다. 기본 정보를 기입해주세요.')

# 음성분석을 위한 개인정보 수집
st.title('음성 분석을 위한 기본정보 입력')

# 개인정보 수집동의
st.write("개인정보 수집 동의")
st.markdown("※ 목소리는 비식별 조치되며, 개인 맞춤 분석 용도로만 사용됩니다.")
personal_data = st.selectbox('개인정보 수집에 동의하십니까?', ['예', '아니요 (※아니요 선택시 개인화 분석 서비스를 이용할 수 없습니다.)'])

# 성별
gender = st.selectbox('성별을 선택하세요', ['남', '여'])

# 국적 선택
nationality = st.selectbox('국적을 선택하세요', ['대한민국', '베트남', '중국', '기타'])

# 연령 선택
age_group = st.selectbox('연령을 선택하세요', ['아동 (변성기 이전)', '청소년', '청년', '장년'])

# 모음별 포먼트 범위 설정
formant_ranges_KOR_man = {
    'ㅏ': {'F1': (600, 850), 'F2': (1100, 1250), 'F3': (2200, 2800)},
    'ㅣ': {'F1': (200, 300), 'F2': (1400, 2200), 'F3': (2500, 3300)},
    'ㅜ': {'F1': (200, 450), 'F2': (600, 1500), 'F3': (2000, 2500)},
    'ㅗ': {'F1': (300, 450), 'F2': (550, 1400), 'F3': (2500, 2900)},
    'ㅡ': {'F1': (300, 500), 'F2': (1200, 1600), 'F3': (2500, 3000)},
    'ㅓ': {'F1': (430, 650), 'F2': (800, 1300), 'F3': (2500, 3000)},
    'ㅐ': {'F1': (400, 520), 'F2': (1300, 2000), 'F3': (2500, 2700)}
}

formant_ranges_KOR_woman = {
    'ㅏ': {'F1': (800, 950), 'F2': (1300, 1800), 'F3': (2600, 3200)},
    'ㅣ': {'F1': (200, 350), 'F2': (1200, 2700), 'F3': (2900, 3200)},
    'ㅜ': {'F1': (330, 440), 'F2': (800, 900), 'F3': (2600, 2900)},
    'ㅗ': {'F1': (360, 460), 'F2': (700, 900), 'F3': (2600, 2900)},
    'ㅡ': {'F1': (380, 450), 'F2': (1500, 1800), 'F3': (2800, 3100)},
    'ㅓ': {'F1': (570, 700), 'F2': (950, 1250), 'F3': (3000, 3300)},
    'ㅐ': {'F1': (540, 650), 'F2': (2200, 2500), 'F3': (3100, 3400)}
}

formant_ranges_CN_man = {
    'ㅏ': {'F1': (800, 870), 'F2': (1300, 1600), 'F3': (2900, 3200)},
    'ㅣ': {'F1': (350, 410), 'F2': (2400, 2900), 'F3': (3500, 3800)},
    'ㅜ': {'F1': (400, 490), 'F2': (800, 900), 'F3': (2600, 2900)},
    'ㅗ': {'F1': (500, 560), 'F2': (700, 900), 'F3': (2600, 2900)},
    'ㅡ': {'F1': (460, 540), 'F2': (1500, 1800), 'F3': (2800, 3100)},
    'ㅓ': {'F1': (600, 650), 'F2': (950, 1250), 'F3': (3000, 3300)},
    'ㅐ': {'F1': (400, 580), 'F2': (1800, 2500), 'F3': (2500, 3400)}
}

formant_ranges_CN_woman = {
    'ㅏ': {'F1': (980, 1290), 'F2': (1350, 1560), 'F3': (2900, 3500)},
    'ㅣ': {'F1': (350, 370), 'F2': (2200, 2800), 'F3': (3500, 4100)},
    'ㅜ': {'F1': (440, 500), 'F2': (910, 1220), 'F3': (2600, 3200)},
    'ㅗ': {'F1': (500, 570), 'F2': (910, 1100), 'F3': (2600, 3200)},
    'ㅡ': {'F1': (480, 680), 'F2': (1500, 1540), 'F3': (2800, 3400)},
    'ㅓ': {'F1': (570, 710), 'F2': (950, 1250), 'F3': (3000, 3700)},
    'ㅐ': {'F1': (670, 750), 'F2': (2200, 2230), 'F3': (3100, 3700)}
}

# 포먼트 추출 함수
def extract_formants(sound, time_step=0.01, max_number_of_formants=5, window_length=0.025, pre_emphasis=50):
    formant = sound.to_formant_burg(time_step=time_step, max_number_of_formants=max_number_of_formants, window_length=window_length, pre_emphasis_from=pre_emphasis)
    formant_values = []
    for t in np.arange(0, sound.get_total_duration(), time_step):
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)
        f3 = formant.get_value_at_time(3, t)
        formant_values.append((t, f1, f2, f3))
    return formant_values

# 특정 모음에 해당하는 포먼트를 필터링하는 함수
def filter_formants_by_vowel(formant_values, vowel):
    if nationality == '대한민국':
        if gender == '남':
            formant_ranges = formant_ranges_KOR_man
        if gender == '여':
            formant_ranges = formant_ranges_KOR_woman
    elif nationality in ['중국', '베트남']:
        if gender == '남':
            formant_ranges = formant_ranges_CN_man
        else:
            formant_ranges = formant_ranges_CN_woman
    else:
        st.write("기타 국적은 현재 지원되지 않습니다.")
        return []
    
    f1_range = formant_ranges[vowel]['F1']
    f2_range = formant_ranges[vowel]['F2']
    f3_range = formant_ranges[vowel]['F3']
    
    filtered_formants = [(t, f1, f2, f3) for t, f1, f2, f3 in formant_values 
                         if f1_range[0] <= f1 <= f1_range[1] and 
                            f2_range[0] <= f2 <= f2_range[1] and 
                            f3_range[0] <= f3 <= f3_range[1]]
    return filtered_formants

# 포먼트 계산
def calculate_average_formants(filtered_formants):
    if not filtered_formants:
        return None, None, None
    f1_values = [f1 for _, f1, _, _ in filtered_formants]
    f2_values = [f2 for _, _, f2, _ in filtered_formants]
    f3_values = [f3 for _, _, _, f3 in filtered_formants]
    
    avg_f1 = np.mean(f1_values)
    avg_f2 = np.mean(f2_values)
    avg_f3 = np.mean(f3_values)
    
    return avg_f1, avg_f2, avg_f3

# Streamlit UI 설정
st.title('실시간 발음정확도 분석')

# 녹음 시간 설정
duration = 3  # seconds

# 모음 선택 UI
vowel = st.selectbox("Select the vowel to analyze", list(formant_ranges_KOR_man.keys()))

if vowel:
    st.subheader(f"가이드 음성: '{vowel}'")
    
    # 성별에 따른 가이드 음성 경로 설정
    if gender == '남':
        audio_file_path = f'https://raw.githubusercontent.com/yourusername/yourrepo/main/Korhub_analyzer/남성/{vowel}.wav'
    else:
        audio_file_path = f'https://raw.githubusercontent.com/yourusername/yourrepo/main/Korhub_analyzer/여성/{vowel}.wav'
    # 가이드 음성 재생
    st.audio(audio_file_path, format='audio/wav')

    st.subheader('실시간 음성 녹음')

# if st.button('Start Recording'):
    # 오디오 녹음
    audio_bytes = audio_recorder()
    
    if audio_bytes:
        # 임시 파일에 저장
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)

        sound = parselmouth.Sound("temp_audio.wav")

        # 포먼트 추출
        formant_values = extract_formants(sound)

        # 선택된 모음에 해당하는 포먼트 필터링
        filtered_formants = filter_formants_by_vowel(formant_values, vowel)

        # 결과 출력
        if filtered_formants:
            avg_f1, avg_f2, avg_f3 = calculate_average_formants(filtered_formants)
            st.write(f"F1: {avg_f1:.2f}Hz, F2: {avg_f2:.2f}Hz, F3: {avg_f3:.2f}Hz")
            
            for t, f1, f2, f3 in filtered_formants:
                st.write(f"Time: {t:.3f}s, F1: {f1:.2f}Hz, F2: {f2:.2f}Hz, F3: {f3:.2f}Hz")
        else:
            st.write(f"해당 발음을 확인할 수 없습니다. '{vowel}' 발음에 유의하여 다시 시도해주세요.")

        # 시각화 (포먼트 그래프)
        if filtered_formants:
            times = [t for t, _, _, _ in filtered_formants]
            f1_values = [f1 for _, f1, _, _ in filtered_formants]
            f2_values = [f2 for _, _, f2, _ in filtered_formants]
            f3_values = [f3 for _, _, _, f3 in filtered_formants]
            
            plt.rcParams['font.family'] = 'AppleGothic'
            plt.figure(figsize=(10, 6))
            plt.plot(times, f1_values, 'r-', label='F1')
            plt.plot(times, f2_values, 'g-', label='F2')
            plt.plot(f3_values, 'b-', label='F3')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.legend()
            plt.title(f'Formant Frequencies for Vowel "{vowel}"')
            st.pyplot(plt)
        
        # 가이드 파형
        st.write("가이드 음성파형")    
        x, fs = librosa.load(audio_file_path, sr=None)
        y = librosa.stft(x, n_fft=128, hop_length=64, win_length=128)

        magnitude = np.abs(y)
        log_spectrogram = librosa.amplitude_to_db(magnitude)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_spectrogram, sr=fs, hop_length=64)
        st.pyplot(plt)

        # 학습자 파형
        st.write("학습자 음성파형")    
        x, fs = librosa.load("temp_audio.wav", sr=None)
        y = librosa.stft(x, n_fft=128, hop_length=64, win_length=128)

        magnitude = np.abs(y)
        log_spectrogram = librosa.amplitude_to_db(magnitude)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_spectrogram, sr=fs, hop_length=64)
        st.pyplot(plt)
