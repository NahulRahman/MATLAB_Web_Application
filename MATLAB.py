import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Signal Converter", layout="wide")

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Function to create a downloadable plot
def get_image_download_link(fig, filename, text):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Title
st.markdown('<div class="main-header">Signal Conversion Tool</div>', unsafe_allow_html=True)

# Main selection
conversion_type = st.selectbox(
    "Select Conversion Type",
    ["Digital to Digital Signal Conversion", 
     "Digital to Analog Signal Conversion",
     "Analog to Analog Signal Conversion", 
     "Analog to Digital Signal Conversion"]
)

# Function for Digital to Digital conversion methods
def digital_to_digital_conversion(data, conversion_method):
    n = len(data)
    t = np.arange(0, n, 0.01)
    
    if conversion_method == "Unipolar NRZ":
        # Unipolar NRZ conversion
        y = np.zeros_like(t)
        for i in range(n):
            idx = (t >= i) & (t < i+1)
            y[idx] = data[i]
        return t, y, "Unipolar NRZ"
    
    elif conversion_method == "Polar NRZ-L (Level)":
        # Polar NRZ-L conversion
        y = np.zeros_like(t)
        for i in range(n):
            idx = (t >= i) & (t < i+1)
            y[idx] = 1 if data[i] == 0 else -1
        return t, y, "Polar NRZ-L (Level)"
    
    elif conversion_method == "Polar NRZI (Inverted)":
        # Polar NRZI conversion
        y = np.zeros_like(t)
        a = 1  # Starting level
        for i in range(n):
            idx = (t >= i) & (t < i+1)
            if data[i] == 1:
                a = -a  # Invert for 1
            y[idx] = a
        return t, y, "Polar NRZI (Inverted)"
    
    elif conversion_method == "Polar RZ (Return to Zero)":
        # Polar RZ conversion
        y = np.zeros_like(t)
        for i in range(n):
            idx1 = (t >= i) & (t < i+0.5)
            idx2 = (t >= i+0.5) & (t < i+1)
            if data[i] == 0:
                y[idx1] = -1
                y[idx2] = 0
            else:
                y[idx1] = 1
                y[idx2] = 0
        return t, y, "Polar RZ (Return to Zero)"
    
    elif conversion_method == "Manchester":
        # Manchester conversion
        y = np.zeros_like(t)
        for i in range(n):
            idx1 = (t >= i) & (t < i+0.5)
            idx2 = (t >= i+0.5) & (t < i+1)
            if data[i] == 0:
                y[idx1] = 1
                y[idx2] = -1
            else:
                y[idx1] = -1
                y[idx2] = 1
        return t, y, "Manchester"
    
    elif conversion_method == "Differential Manchester":
        # Differential Manchester conversion
        y = np.zeros_like(t)
        a = 1  # Starting level
        for i in range(n):
            idx1 = (t >= i) & (t < i+0.5)
            idx2 = (t >= i+0.5) & (t < i+1)
            if data[i] == 0:
                y[idx1] = -a
                y[idx2] = a
            else:
                y[idx1] = a
                y[idx2] = -a
            a = y[idx2][0] if len(y[idx2]) > 0 else a
        return t, y, "Differential Manchester"
    
    elif conversion_method == "Bipolar AMI":
        # Bipolar AMI conversion
        y = np.zeros_like(t)
        a = 1  # Starting polarity
        for i in range(n):
            idx = (t >= i) & (t < i+1)
            if data[i] == 1:
                y[idx] = a
                a = -a  # Alternate polarity for next '1'
            else:
                y[idx] = 0
        return t, y, "Bipolar AMI"
    
    elif conversion_method == "Bipolar Pseudoternary":
        # Bipolar Pseudoternary conversion
        y = np.zeros_like(t)
        a = 1  # Starting polarity
        for i in range(n):
            idx = (t >= i) & (t < i+1)
            if data[i] == 0:
                y[idx] = a
                a = -a  # Alternate polarity for next '0'
            else:
                y[idx] = 0
        return t, y, "Bipolar Pseudoternary"
    
    elif conversion_method == "MLT-3":
        # MLT-3 conversion
        y = np.zeros_like(t)
        cur_lvl = 0
        last_non_zero_lvl = -1
        for i in range(n):
            idx = (t >= i) & (t < i+1)
            if data[i] == 0:
                y[idx] = cur_lvl
            else:
                if cur_lvl != 0:
                    y[idx] = 0
                    cur_lvl = 0
                else:
                    last_non_zero_lvl = -last_non_zero_lvl
                    y[idx] = last_non_zero_lvl
                    cur_lvl = last_non_zero_lvl
        return t, y, "MLT-3"
    
    return None, None, None

# Function for Digital to Analog conversion methods
def digital_to_analog_conversion(data, conversion_method, parameters):
    n = len(data)
    bp = parameters.get('bp', 0.00001)
    bit = []
    
    # Create bit representation
    for i in range(n):
        if data[i] == 1:
            se = np.zeros(100)
        else:
            se = np.ones(100)
        bit = np.append(bit, se)
    
    t1 = np.arange(bp/100, bp*n + bp/100, bp/100)
    
    if conversion_method == "ASK":
        # Amplitude Shift Keying
        A0 = parameters.get('A0', 0)
        A1 = parameters.get('A1', 1)
        f = parameters.get('f', 5/bp)
        st = []
        t2 = np.arange(bp/100, bp + bp/100, bp/100)
        
        for i in range(n):
            if data[i] == 1:
                y = A1 * np.sin(2 * np.pi * f * t2)
            else:
                y = A0 * np.sin(2 * np.pi * f * t2)
            st = np.append(st, y)
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(t1, bit)
        axs[0].set_title('Original Digital Signal')
        axs[0].set_xlim(0, n*bp)
        axs[0].set_ylim(-1.5, 1.5)
        axs[0].grid(True)
        
        axs[1].plot(t1, st)
        axs[1].set_title('ASK Modulated Signal')
        axs[1].set_xlim(0, n*bp)
        axs[1].set_ylim(-1.5, 1.5)
        axs[1].grid(True)
        
        plt.tight_layout()
        return fig, "ASK Modulation"
    
    elif conversion_method == "FSK":
        # Frequency Shift Keying
        A = parameters.get('A', 1)
        f1 = parameters.get('f1', 5/bp)
        f0 = parameters.get('f0', 2/bp)
        st = []
        t2 = np.arange(bp/100, bp + bp/100, bp/100)
        
        for i in range(n):
            if data[i] == 1:
                y = A * np.sin(2 * np.pi * f1 * t2)
            else:
                y = A * np.sin(2 * np.pi * f0 * t2)
            st = np.append(st, y)
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(t1, bit)
        axs[0].set_title('Original Digital Signal')
        axs[0].set_xlim(0, n*bp)
        axs[0].set_ylim(-1.5, 1.5)
        axs[0].grid(True)
        
        axs[1].plot(t1, st)
        axs[1].set_title('FSK Modulated Signal')
        axs[1].set_xlim(0, n*bp)
        axs[1].set_ylim(-1.5, 1.5)
        axs[1].grid(True)
        
        plt.tight_layout()
        return fig, "FSK Modulation"
    
    elif conversion_method == "PSK":
        # Phase Shift Keying
        A = parameters.get('A', 1)
        f = parameters.get('f', 5/bp)
        st = []
        t2 = np.arange(bp/100, bp + bp/100, bp/100)
        
        for i in range(n):
            if data[i] == 1:
                y = -A * np.cos(2 * np.pi * f * t2)
            else:
                y = -A * np.sin(2 * np.pi * f * t2)
            st = np.append(st, y)
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(t1, bit)
        axs[0].set_title('Original Digital Signal')
        axs[0].set_xlim(0, n*bp)
        axs[0].set_ylim(-1.5, 1.5)
        axs[0].grid(True)
        
        axs[1].plot(t1, st)
        axs[1].set_title('PSK Modulated Signal')
        axs[1].set_xlim(0, n*bp)
        axs[1].set_ylim(-1.5, 1.5)
        axs[1].grid(True)
        
        plt.tight_layout()
        return fig, "PSK Modulation"
    
    return None, None

# Function for Analog to Analog conversion methods
def analog_to_analog_conversion(conversion_method, parameters):
    # Time parameters
    fs = parameters.get('fs', 1000)  # Sampling frequency (Hz)
    t = np.linspace(0, 1, fs)  # Time vector (0 to 1 second)
    
    # Message signal parameters
    fm = parameters.get('fm', 10)  # Message signal frequency (Hz)
    Am = parameters.get('Am', 1)  # Message signal amplitude
    message_signal = Am * np.sin(2 * np.pi * fm * t)  # Message signal
    
    # Carrier signal parameters
    fc = parameters.get('fc', 100)  # Carrier signal frequency (Hz)
    Ac = parameters.get('Ac', 2)  # Carrier signal amplitude
    carrier_signal = Ac * np.sin(2 * np.pi * fc * t)  # Carrier signal
    
    if conversion_method == "Amplitude Modulation":
        # Modulated signal
        modulated_signal = (1 + message_signal) * carrier_signal
        
        # Demodulation
        # Rectification process
        rectified_signal = np.abs(modulated_signal)
        
        # Low pass filter parameters
        cutoff_freq = parameters.get('cutoff_freq', 20)
        
        # Design the low pass filter
        order = 4  # Filter order
        nyquist_freq = fs/2  # Hz
        normalized_cutoff_freq = cutoff_freq / nyquist_freq
        b, a = signal.butter(order, normalized_cutoff_freq, 'low')
        
        # Apply the lowpass filter to the rectified signal
        filtered_signal = signal.filtfilt(b, a, rectified_signal)
        
        fig, axs = plt.subplots(5, 1, figsize=(10, 15))
        axs[0].plot(t, message_signal)
        axs[0].set_title('Message Signal')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True)
        
        axs[1].plot(t, carrier_signal)
        axs[1].set_title('Carrier Signal')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Amplitude')
        axs[1].grid(True)
        
        axs[2].plot(t, modulated_signal)
        axs[2].set_title('Modulated Signal')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Amplitude')
        axs[2].grid(True)
        
        axs[3].plot(t, rectified_signal)
        axs[3].set_title('Rectified Signal')
        axs[3].set_xlabel('Time (s)')
        axs[3].set_ylabel('Amplitude')
        axs[3].grid(True)
        
        axs[4].plot(t, filtered_signal)
        axs[4].set_title('Filtered Signal (Demodulated)')
        axs[4].set_xlabel('Time (s)')
        axs[4].set_ylabel('Amplitude')
        axs[4].grid(True)
        
        plt.tight_layout()
        return fig, "Amplitude Modulation and Demodulation"
    
    elif conversion_method == "Frequency Modulation":
        # Frequency Modulation parameters
        kf = parameters.get('kf', 10)  # Frequency deviation constant
        
        # Modulated signal
        modulated_signal = Ac * np.sin(2 * np.pi * fc * t + kf * message_signal)
        
        # Demodulation
        # Use differentiation to extract the frequency changes
        demodulated_signal = np.diff(modulated_signal) * fs / kf
        # Add a zero to make it the same length as other signals for plotting
        demodulated_signal = np.append(demodulated_signal, demodulated_signal[-1])
        
        fig, axs = plt.subplots(4, 1, figsize=(10, 12))
        axs[0].plot(t, message_signal)
        axs[0].set_title('Message Signal')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True)
        
        axs[1].plot(t, carrier_signal)
        axs[1].set_title('Carrier Signal')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Amplitude')
        axs[1].grid(True)
        
        axs[2].plot(t, modulated_signal)
        axs[2].set_title('Frequency Modulated Signal')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Amplitude')
        axs[2].grid(True)
        
        axs[3].plot(t, demodulated_signal)
        axs[3].set_title('Demodulated Signal')
        axs[3].set_xlabel('Time (s)')
        axs[3].set_ylabel('Amplitude')
        axs[3].grid(True)
        
        plt.tight_layout()
        return fig, "Frequency Modulation and Demodulation"
    
    return None, None

# Function for Analog to Digital conversion methods
def analog_to_digital_conversion(conversion_method, parameters):
    if conversion_method == "Delta Modulation and demodulation":
        # Time parameters
        fs = parameters.get('fs', 10000)
        fm = parameters.get('fm', 100)
        duration = parameters.get('duration', 0.1)  # Time Duration in seconds
        t = np.arange(0, duration, 1/fs)
        
        # Original analog signal
        amplitude = parameters.get('amplitude', 5)
        x = amplitude * np.sin(2 * np.pi * fm * t)
        
        # Delta modulation parameters
        level = parameters.get('level', 0.4)  # Step size
        
        # Perform delta modulation
        y = np.zeros_like(t)
        xr = np.zeros_like(t)  # Staircase approximation
        
        for i in range(len(t)-1):
            if xr[i] <= x[i]:
                y[i+1] = 1
                xr[i+1] = xr[i] + level
            else:
                y[i+1] = 0
                xr[i+1] = xr[i] - level
        
        # Calculate Mean Squared Error (MSE)
        mse = np.sum((x - xr)**2) / len(x)
        
        # Delta Demodulation
        y_demod = np.zeros_like(t)
        xr_demod = 0
        
        for i in range(1, len(t)):
            if y[i] == 1:
                xr_demod += level
            else:
                xr_demod -= level
            y_demod[i] = xr_demod
        
        # Apply Low-Pass Filter for demodulation
        filter_order = 20
        cutoff = fm/(fs/2)
        lowpass_filter = signal.firwin(filter_order+1, cutoff, window='hamming')
        filtered_demod_signal = signal.filtfilt(lowpass_filter, [1], y_demod)
        
        fig, axs = plt.subplots(4, 1, figsize=(10, 12))
        
        axs[0].plot(t, x)
        axs[0].set_title('Original Analog Signal')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True)
        
        axs[1].step(t, xr)
        axs[1].set_title(f'Staircase Approximated Signal (MSE: {mse:.6f})')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Amplitude')
        axs[1].grid(True)
        
        axs[2].step(t, y)
        axs[2].set_title('Delta Modulated Signal (Binary)')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Amplitude')
        axs[2].grid(True)
        
        axs[3].plot(t, filtered_demod_signal)
        axs[3].set_title('Filtered Demodulated Signal')
        axs[3].set_xlabel('Time (s)')
        axs[3].set_ylabel('Amplitude')
        axs[3].grid(True)
        
        plt.tight_layout()
        return fig, "Delta Modulation and Demodulation"
    
    return None, None

# Conversion subtype based on main conversion type
if conversion_type == "Digital to Digital Signal Conversion":
    conversion_subtype = st.selectbox(
        "Select Digital to Digital Conversion Method",
        ["Unipolar NRZ", "Polar NRZ-L (Level)", "Polar NRZI (Inverted)", 
         "Polar RZ (Return to Zero)", "Manchester", "Differential Manchester", 
         "Bipolar AMI", "Bipolar Pseudoternary", "MLT-3"]
    )
    
    # Input for digital data
    st.markdown('<div class="sub-header">Input Parameters</div>', unsafe_allow_html=True)
    input_method = st.radio("Select input method", ["Binary String", "Bit Array"])
    
    if input_method == "Binary String":
        binary_string = st.text_input("Enter binary data (e.g., '10110010')", "10110010")
        data = np.array([int(bit) for bit in binary_string])
    else:
        bit_array = st.text_input("Enter bit array (e.g., '1,0,1,1,0,0,1,0')", "1,0,1,1,0,0,1,0")
        data = np.array([int(bit) for bit in bit_array.split(',')])
    
    # Display binary data
    st.write(f"Input Data: {data}")
    
    if st.button("Generate Digital Signal"):
        t, y, title = digital_to_digital_conversion(data, conversion_subtype)
        
        if t is not None and y is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(t, y, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.set_ylim(-2, 2)
            ax.grid(True)
            
            st.pyplot(fig)
            
            # Provide download link
            st.markdown(get_image_download_link(fig, f"{title.replace(' ', '_')}.png", 
                                                "Download Plot as PNG"), unsafe_allow_html=True)

elif conversion_type == "Digital to Analog Signal Conversion":
    conversion_subtype = st.selectbox(
        "Select Digital to Analog Conversion Method",
        ["ASK", "FSK", "PSK"]
    )
    
    # Input for digital data
    st.markdown('<div class="sub-header">Input Parameters</div>', unsafe_allow_html=True)
    binary_string = st.text_input("Enter binary data (e.g., '10110010')", "10110010")
    data = np.array([int(bit) for bit in binary_string])
    
    # Parameters based on conversion type
    if conversion_subtype == "ASK":
        col1, col2, col3 = st.columns(3)
        with col1:
            bp = st.number_input("Bit Period (bp)", value=0.00001, format="%.8f")
        with col2:
            A0 = st.number_input("Amplitude for 0 (A0)", value=0.0, step=0.1)
        with col3:
            A1 = st.number_input("Amplitude for 1 (A1)", value=1.0, step=0.1)
        frequency = st.slider("Carrier Frequency (relative to 1/bp)", 1, 10, 5)
        parameters = {'bp': bp, 'A0': A0, 'A1': A1, 'f': frequency/bp}
    
    elif conversion_subtype == "FSK":
        col1, col2, col3 = st.columns(3)
        with col1:
            bp = st.number_input("Bit Period (bp)", value=0.00001, format="%.8f")
        with col2:
            A = st.number_input("Amplitude (A)", value=1.0, step=0.1)
        with col3:
            f1 = st.slider("Frequency for 1 (relative to 1/bp)", 3, 10, 5)
        f0 = st.slider("Frequency for 0 (relative to 1/bp)", 1, f1-1, 2)
        parameters = {'bp': bp, 'A': A, 'f1': f1/bp, 'f0': f0/bp}
    
    elif conversion_subtype == "PSK":
        col1, col2 = st.columns(2)
        with col1:
            bp = st.number_input("Bit Period (bp)", value=0.00001, format="%.8f")
        with col2:
            A = st.number_input("Amplitude (A)", value=1.0, step=0.1)
        frequency = st.slider("Carrier Frequency (relative to 1/bp)", 1, 10, 5)
        parameters = {'bp': bp, 'A': A, 'f': frequency/bp}
    
    if st.button("Generate Modulated Signal"):
        fig, title = digital_to_analog_conversion(data, conversion_subtype, parameters)
        
        if fig is not None:
            st.pyplot(fig)
            
            # Provide download link
            st.markdown(get_image_download_link(fig, f"{title.replace(' ', '_')}.png", 
                                                "Download Plot as PNG"), unsafe_allow_html=True)

elif conversion_type == "Analog to Analog Signal Conversion":
    conversion_subtype = st.selectbox(
        "Select Analog to Analog Conversion Method",
        ["Amplitude Modulation", "Frequency Modulation"]
    )
    
    # Common parameters
    st.markdown('<div class="sub-header">Signal Parameters</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fs = st.number_input("Sampling Frequency (Hz)", value=1000, step=100)
    with col2:
        Am = st.number_input("Message Signal Amplitude", value=1.0, step=0.1)
    with col3:
        fm = st.number_input("Message Signal Frequency (Hz)", value=10, step=1)
    
    col1, col2 = st.columns(2)
    with col1:
        Ac = st.number_input("Carrier Signal Amplitude", value=2.0, step=0.1)
    with col2:
        fc = st.number_input("Carrier Signal Frequency (Hz)", value=100, step=10)
    
    parameters = {'fs': fs, 'Am': Am, 'fm': fm, 'Ac': Ac, 'fc': fc}
    
    # Parameters specific to modulation type
    if conversion_subtype == "Amplitude Modulation":
        cutoff_freq = st.slider("Demodulation Cutoff Frequency (Hz)", 5, 50, 20)
        parameters['cutoff_freq'] = cutoff_freq
    
    elif conversion_subtype == "Frequency Modulation":
        kf = st.slider("Frequency Deviation Constant", 1, 50, 10)
        parameters['kf'] = kf
    
    if st.button("Generate Modulated Signal"):
        fig, title = analog_to_analog_conversion(conversion_subtype, parameters)
        
        if fig is not None:
            st.pyplot(fig)
            
            # Provide download link
            st.markdown(get_image_download_link(fig, f"{title.replace(' ', '_')}.png", 
                                                "Download Plot as PNG"), unsafe_allow_html=True)

elif conversion_type == "Analog to Digital Signal Conversion":
    conversion_subtype = "Delta Modulation and demodulation"  # Only one option available
    st.write(f"Selected: {conversion_subtype}")
    
    # Parameters for Delta Modulation
    st.markdown('<div class="sub-header">Signal Parameters</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fs = st.number_input("Sampling Frequency (Hz)", value=10000, step=1000)
    with col2:
        fm = st.number_input("Message Signal Frequency (Hz)", value=100, step=10)
    with col3:
        amplitude = st.number_input("Signal Amplitude", value=5.0, step=0.5)
    
    col1, col2 = st.columns(2)
    with col1:
        duration = st.number_input("Signal Duration (seconds)", value=0.1, step=0.01, format="%.3f")
    with col2:
        level = st.number_input("Step Size (Delta)", value=0.4, step=0.1, format="%.2f")
    
    parameters = {'fs': fs, 'fm': fm, 'amplitude': amplitude, 'duration': duration, 'level': level}
    
    if st.button("Generate Delta Modulation"):
        fig, title = analog_to_digital_conversion(conversion_subtype, parameters)
        
        if fig is not None:
            st.pyplot(fig)
            
            # Provide download link
            st.markdown(get_image_download_link(fig, f"{title.replace(' ', '_')}.png", 
                                                "Download Plot as PNG"), unsafe_allow_html=True)

# Footer with info
st.markdown("---")
st.markdown("Signal Conversion Tool - Convert between various signal types and visualize waveforms")