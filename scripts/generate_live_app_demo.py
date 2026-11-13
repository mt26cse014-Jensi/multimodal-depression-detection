import math
import os
import random
from PIL import Image, ImageDraw

def create_live_app_demo_gif(output_path="outputs/plots/live_app_demo.gif", num_frames=36):
    width, height = 1000, 580
    frames = []

    # Palette
    c_bg = (11, 15, 25)
    c_panel_bg = (18, 24, 42)
    c_panel_border = (38, 50, 85)
    c_cyan = (0, 243, 255)
    c_magenta = (255, 0, 127)
    c_purple = (150, 80, 255)
    c_amber = (255, 170, 0)
    c_emerald = (0, 255, 150)
    c_red = (255, 65, 85)
    c_text_bright = (245, 248, 255)
    c_text_dim = (130, 150, 185)

    tokens = [
        "I", "feel", "really", "drained", "lately,", "like", "getting", "out",
        "of", "bed", "is", "a", "huge", "struggle", "every", "morning...",
        "I", "used", "to", "enjoy", "reading", "and", "walking,", "but",
        "now", "nothing", "really", "brings", "me", "joy", "anymore."
    ]

    for frame_idx in range(num_frames):
        t = frame_idx / num_frames
        im = Image.new("RGB", (width, height), c_bg)
        draw = ImageDraw.Draw(im)

        # 1. Window Frame & Title Bar
        draw.rounded_rectangle([20, 15, width - 20, height - 15], radius=12, fill=c_bg, outline=c_panel_border, width=2)
        draw.rounded_rectangle([20, 15, width - 20, 55], radius=12, fill=(24, 32, 56))
        # Window buttons (Mac style)
        draw.ellipse([35, 30, 47, 42], fill=(255, 95, 87))
        draw.ellipse([55, 30, 67, 42], fill=(254, 188, 46))
        draw.ellipse([75, 30, 87, 42], fill=(40, 200, 64))

        # Title text
        draw.text((110, 28), "Multimodal Clinical Inference Engine v1.0 [Session: DAIC-Participant-314]", fill=c_text_bright)
        draw.text((width - 240, 28), "● REALTIME EVALUATION", fill=c_emerald)

        # 2. Panel 1: Live Audio Stream (Top Left)
        p1 = (35, 70, 485, 290)
        draw.rounded_rectangle(p1, radius=8, fill=c_panel_bg, outline=c_panel_border, width=1)
        draw.text((p1[0] + 16, p1[1] + 12), "ACOUSTIC SIGNAL (16 kHz Mono)", fill=c_cyan)
        draw.text((p1[0] + 280, p1[1] + 12), "Wav2Vec2 / Turn-Moments", fill=c_text_dim)

        # Waveform oscilloscope animation
        wave_y_mid = p1[1] + 105
        wave_pts = []
        for x_idx, x in enumerate(range(p1[0] + 20, p1[2] - 20, 4)):
            # Simulated speech envelope with harmonic oscillations
            envelope = math.sin((x_idx / 8.0) + (t * 4 * math.pi)) * math.exp(-((x_idx % 25 - 12)**2) / 60.0)
            voice_mod = math.sin(x_idx * 0.8 + t * 8 * math.pi) * 28 * abs(envelope)
            wave_pts.append((x, wave_y_mid + voice_mod))

        for i in range(len(wave_pts) - 1):
            draw.line([wave_pts[i], wave_pts[i+1]], fill=c_cyan, width=2)

        # Acoustic stats bar
        draw.text((p1[0] + 20, p1[1] + 170), f"F0 Mean: {128.4 + 4*math.sin(t*6.28):.1f} Hz", fill=c_text_bright)
        draw.text((p1[0] + 170, p1[1] + 170), f"Jitter: {0.024 + 0.005*math.cos(t*6.28):.3f}", fill=c_text_bright)
        draw.text((p1[0] + 300, p1[1] + 170), "Voice Tenseness: High", fill=c_amber)

        # Mini Spectrogram bar
        spec_y = p1[1] + 195
        draw.text((p1[0] + 20, spec_y), "MFCC 1-13 Energy Spectrum:", fill=c_text_dim)
        for b_idx in range(13):
            bar_h = int(12 + 10 * math.sin(b_idx * 0.6 + t * 7))
            bx = p1[0] + 20 + b_idx * 32
            draw.rectangle([bx, spec_y + 20 - bar_h, bx + 22, spec_y + 20], fill=c_purple)

        # 3. Panel 2: Live Transcript & Attention (Top Right)
        p2 = (505, 70, width - 35, 290)
        draw.rounded_rectangle(p2, radius=8, fill=c_panel_bg, outline=c_panel_border, width=1)
        draw.text((p2[0] + 16, p2[1] + 12), "TRANSCRIPT & ATTENTION HIGHLIGHTING", fill=c_magenta)
        draw.text((p2[0] + 320, p2[1] + 12), "MentalBERT [CLS]", fill=c_text_dim)

        # Animated token stream
        visible_token_count = min(len(tokens), int(12 + t * (len(tokens) - 8)))
        curr_x, curr_y = p2[0] + 16, p2[1] + 45
        for tok_idx in range(visible_token_count):
            tok = tokens[tok_idx]
            # Highlight high-importance clinical tokens
            is_critical = tok in ["drained", "struggle", "joy", "nothing"]
            tok_color = c_magenta if is_critical else c_text_bright
            bg_box = is_critical and (t > 0.3)

            tok_width = len(tok) * 8 + 8
            if curr_x + tok_width > p2[2] - 20:
                curr_x = p2[0] + 16
                curr_y += 28

            if bg_box:
                draw.rectangle([curr_x - 2, curr_y - 2, curr_x + tok_width - 4, curr_y + 16], fill=(50, 15, 40))
            draw.text((curr_x, curr_y), tok, fill=tok_color)
            curr_x += tok_width + 4

        draw.text((p2[0] + 16, p2[1] + 185), "Top Attention Token: [drained] (weight: 0.342)", fill=c_magenta)
        draw.text((p2[0] + 16, p2[1] + 205), "Contextual Valence: Negative Affective Polarity", fill=c_text_dim)

        # 4. Panel 3: Multi-Task Diagnostic Prediction (Bottom Left)
        p3 = (35, 310, 485, height - 30)
        draw.rounded_rectangle(p3, radius=8, fill=c_panel_bg, outline=c_panel_border, width=1)
        draw.text((p3[0] + 16, p3[1] + 12), "MULTI-TASK DIAGNOSTIC PREDICTIONS", fill=c_emerald)

        # Depression Probability Meter
        prob_val = min(0.78, 0.25 + 0.53 * min(1.0, t * 1.5))
        draw.text((p3[0] + 20, p3[1] + 45), f"Depression Probability: {prob_val * 100:.1f}%", fill=c_text_bright)

        # Progress bar
        bar_x1, bar_y1, bar_x2, bar_y2 = p3[0] + 20, p3[1] + 70, p3[2] - 20, p3[1] + 90
        draw.rectangle([bar_x1, bar_y1, bar_x2, bar_y2], fill=(30, 40, 65))
        fill_x = bar_x1 + int((bar_x2 - bar_x1) * prob_val)
        meter_col = c_amber if prob_val < 0.50 else c_red
        draw.rectangle([bar_x1, bar_y1, fill_x, bar_y2], fill=meter_col)

        # Classification Badge
        badge_text = "STATUS: ELEVATED DEPRESSION RISK" if prob_val >= 0.50 else "STATUS: SCREENING NEGATIVE"
        draw.rectangle([p3[0] + 20, p3[1] + 105, p3[0] + 310, p3[1] + 130], fill=(45, 20, 35) if prob_val >= 0.50 else (20, 40, 35))
        draw.text((p3[0] + 30, p3[1] + 110), badge_text, fill=c_red if prob_val >= 0.50 else c_emerald)

        # Continuous PHQ-8 Regression Prediction
        phq_score = min(14.2, 4.0 + 10.2 * min(1.0, t * 1.5))
        draw.text((p3[0] + 20, p3[1] + 150), f"Predicted PHQ-8 Score: {phq_score:.1f} / 24", fill=c_text_bright)
        draw.text((p3[0] + 20, p3[1] + 172), "Clinical Severity: Moderate-to-Severe Range (Cutoff >= 10)", fill=c_amber)
        draw.text((p3[0] + 20, p3[1] + 195), "Participant Speech Filter: Ellie Interviewer Speech REMOVED (+0.082 F1)", fill=c_cyan)

        # 5. Panel 4: Explainable AI & SHAP Attributions (Bottom Right)
        p4 = (505, 310, width - 35, height - 30)
        draw.rounded_rectangle(p4, radius=8, fill=c_panel_bg, outline=c_panel_border, width=1)
        draw.text((p4[0] + 16, p4[1] + 12), "EXPLAINABILITY & MODALITY ATTRIBUTION (SHAP)", fill=c_purple)

        # SHAP Modality Bar
        draw.text((p4[0] + 20, p4[1] + 45), "KernelSHAP Modality Contribution Ratio:", fill=c_text_bright)
        m_bar_x1, m_bar_y1, m_bar_x2, m_bar_y2 = p4[0] + 20, p4[1] + 70, p4[2] - 20, p4[1] + 90
        draw.rectangle([m_bar_x1, m_bar_y1, m_bar_x2, m_bar_y2], fill=(30, 40, 65))
        # Text 79%, Audio 21%
        split_x = m_bar_x1 + int((m_bar_x2 - m_bar_x1) * 0.792)
        draw.rectangle([m_bar_x1, m_bar_y1, split_x, m_bar_y2], fill=c_magenta)
        draw.rectangle([split_x, m_bar_y1, m_bar_x2, m_bar_y2], fill=c_cyan)

        draw.text((p4[0] + 20, p4[1] + 100), "■ Linguistic (Text): 79.2%", fill=c_magenta)
        draw.text((p4[0] + 220, p4[1] + 100), "■ Acoustic (Audio): 20.8%", fill=c_cyan)

        draw.text((p4[0] + 20, p4[1] + 135), "Key Biomarkers Identified:", fill=c_text_bright)
        draw.text((p4[0] + 20, p4[1] + 155), "1. Persistent lexical anhedonia & self-reported exhaustion", fill=c_text_dim)
        draw.text((p4[0] + 20, p4[1] + 175), "2. Reduced pitch dynamicity (flat vocal tone in mid-registers)", fill=c_text_dim)
        draw.text((p4[0] + 20, p4[1] + 195), "Interview Turn: 48 / 112 (Exploration Phase • Non-Shortcut)", fill=c_emerald)

        frames.append(im)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=70,  # ~14 fps
        loop=0,
        optimize=True
    )
    print(f"Generated {output_path} ({len(frames)} frames, {os.path.getsize(output_path)} bytes)")

if __name__ == "__main__":
    create_live_app_demo_gif()
