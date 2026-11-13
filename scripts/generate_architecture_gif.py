import math
import os
from PIL import Image, ImageDraw, ImageFont

def create_system_architecture_gif(output_path="outputs/plots/system_architecture.gif", num_frames=30):
    width, height = 1000, 560
    frames = []

    # Modern dark cyberpunk color palette
    c_bg = (10, 14, 28)
    c_card_bg = (18, 25, 48)
    c_card_border = (40, 55, 95)
    c_cyan = (0, 243, 255)
    c_purple = (180, 80, 255)
    c_magenta = (255, 40, 140)
    c_amber = (255, 180, 30)
    c_emerald = (0, 255, 170)
    c_text_bright = (240, 245, 255)
    c_text_dim = (140, 160, 195)
    c_line_dim = (30, 45, 80)

    # Box coordinates [x1, y1, x2, y2]
    # Col 1: Raw Inputs
    b_audio = (50, 110, 230, 210)
    b_text = (50, 310, 230, 410)

    # Col 2: Pretrained Encoders
    b_enc_audio = (290, 110, 470, 210)
    b_enc_text = (290, 310, 470, 410)

    # Col 3: Cross-Modal Fusion
    b_fusion = (530, 180, 720, 340)

    # Col 4: Multi-Task Heads
    b_head_cls = (780, 140, 950, 230)
    b_head_reg = (780, 290, 950, 380)

    # Flow paths for animated signal pulses: list of [(x1, y1), (x2, y2), ...]
    path_audio_to_enc = [(230, 160), (290, 160)]
    path_text_to_enc = [(230, 360), (290, 360)]

    path_enc_audio_to_fuse = [(470, 160), (500, 160), (500, 230), (530, 230)]
    path_enc_text_to_fuse = [(470, 360), (500, 360), (500, 290), (530, 290)]

    path_fuse_to_cls = [(720, 230), (750, 230), (750, 185), (780, 185)]
    path_fuse_to_reg = [(720, 290), (750, 290), (750, 335), (780, 335)]

    all_paths = [
        (path_audio_to_enc, c_cyan),
        (path_text_to_enc, c_magenta),
        (path_enc_audio_to_fuse, c_cyan),
        (path_enc_text_to_fuse, c_magenta),
        (path_fuse_to_cls, c_emerald),
        (path_fuse_to_reg, c_amber),
    ]

    def get_path_length(path):
        length = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += math.hypot(dx, dy)
        return length

    def get_point_along_path(path, progress):
        total_len = get_path_length(path)
        target_dist = (progress % 1.0) * total_len
        curr_dist = 0
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if curr_dist + seg_len >= target_dist:
                seg_prog = (target_dist - curr_dist) / max(seg_len, 1e-5)
                return (
                    p1[0] + seg_prog * (p2[0] - p1[0]),
                    p1[1] + seg_prog * (p2[1] - p1[1])
                )
            curr_dist += seg_len
        return path[-1]

    for frame_idx in range(num_frames):
        t = frame_idx / num_frames
        im = Image.new("RGB", (width, height), c_bg)
        draw = ImageDraw.Draw(im)

        # 1. Subtle Background Grid
        for gx in range(40, width, 40):
            draw.line([(gx, 0), (gx, height)], fill=(15, 20, 38), width=1)
        for gy in range(40, height, 40):
            draw.line([(0, gy), (width, gy)], fill=(15, 20, 38), width=1)

        # 2. Section Header
        draw.text((width // 2 - 210, 30), "MULTIMODAL DEPRESSION DETECTION PIPELINE", fill=c_text_bright)
        draw.text((width // 2 - 235, 55), "Acoustic-Textual Encoders -> Cross-Modal Attention -> Multi-Task Heads", fill=c_text_dim)

        # 3. Draw Static Connection Lines
        for path, col in all_paths:
            for i in range(len(path) - 1):
                draw.line([path[i], path[i+1]], fill=c_line_dim, width=3)

        # 4. Draw Animated Pulses Along Paths
        for path, col in all_paths:
            # 2 pulses per path spaced 0.5 apart
            for offset in [0.0, 0.5]:
                pos = get_point_along_path(path, t + offset)
                # Outer glow
                draw.ellipse([pos[0]-6, pos[1]-6, pos[0]+6, pos[1]+6], fill=col)
                # Bright core
                draw.ellipse([pos[0]-3, pos[1]-3, pos[0]+3, pos[1]+3], fill=(255, 255, 255))

        # Helper to draw rounded card
        def draw_card(box, title, subtitle, badge, accent_col):
            x1, y1, x2, y2 = box
            # Card background
            draw.rounded_rectangle([x1, y1, x2, y2], radius=10, fill=c_card_bg, outline=c_card_border, width=2)
            # Accent bar top
            draw.rounded_rectangle([x1, y1, x2, y1+4], radius=2, fill=accent_col)
            # Text
            draw.text((x1 + 14, y1 + 14), title, fill=c_text_bright)
            draw.text((x1 + 14, y1 + 38), subtitle, fill=c_text_dim)
            if badge:
                draw.rectangle([x1 + 14, y2 - 28, x2 - 14, y2 - 10], fill=(25, 35, 65))
                draw.text((x1 + 20, y2 - 26), badge, fill=accent_col)

        # 5. Draw Module Cards
        draw_card(b_audio, "Acoustic Input", "16 kHz Mono Speech", "EATD / DAIC-WOZ", c_cyan)
        draw_card(b_text, "Linguistic Input", "Participant Transcripts", "De-identified Tokens", c_magenta)

        draw_card(b_enc_audio, "Audio Encoders", "Wav2Vec2 XLS-R (1024d)", "FROZEN PRETRAINED", c_cyan)
        draw_card(b_enc_text, "Text Encoders", "MentalBERT (768d)", "FROZEN PRETRAINED", c_magenta)

        # Fusion Card (with animated glowing border)
        pulse_val = 0.5 + 0.5 * math.sin(2 * math.pi * t)
        pulse_col = (
            int(c_purple[0] * pulse_val + c_card_border[0] * (1 - pulse_val)),
            int(c_purple[1] * pulse_val + c_card_border[1] * (1 - pulse_val)),
            int(c_purple[2] * pulse_val + c_card_border[2] * (1 - pulse_val)),
        )
        draw.rounded_rectangle(b_fusion, radius=12, fill=(24, 32, 60), outline=pulse_col, width=3)
        draw.text((b_fusion[0] + 16, b_fusion[1] + 16), "Multimodal Fusion Engine", fill=c_text_bright)
        draw.text((b_fusion[0] + 16, b_fusion[1] + 40), "Cross-Modal Multi-Head Attention", fill=c_purple)
        draw.text((b_fusion[0] + 16, b_fusion[1] + 62), "• Text attends to Audio Frames", fill=c_text_dim)
        draw.text((b_fusion[0] + 16, b_fusion[1] + 82), "• Audio attends to Text Tokens", fill=c_text_dim)
        draw.text((b_fusion[0] + 16, b_fusion[1] + 104), "• Bidirectional Cross-Attn (4 Heads)", fill=c_text_dim)
        draw.rectangle([b_fusion[0] + 16, b_fusion[3] - 28, b_fusion[2] - 16, b_fusion[3] - 10], fill=(35, 45, 80))
        draw.text((b_fusion[0] + 22, b_fusion[3] - 26), "d_model=128 | 494k params", fill=c_cyan)

        # Output Heads
        draw_card(b_head_cls, "Binary Classification", "Depression Screening", "Macro F1: 0.686", c_emerald)
        draw_card(b_head_reg, "Severity Regression", "Continuous PHQ-8 Score", "RMSE: 6.168", c_amber)

        # 6. Bottom Status Footer
        draw.text((50, height - 35), "Status: 70/70 Unit Tests Passed  |  Bootstrap 95% CI Verified  |  Participant-Level Isolation", fill=(90, 110, 150))
        draw.text((width - 250, height - 35), "Research Prototype (v1.0)", fill=(90, 110, 150))

        frames.append(im)

    # Save as high-quality optimized animated GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,  # 50 ms = 20 fps
        loop=0,
        optimize=True
    )
    print(f"Generated {output_path} ({len(frames)} frames, {os.path.getsize(output_path)} bytes)")

if __name__ == "__main__":
    create_system_architecture_gif()
