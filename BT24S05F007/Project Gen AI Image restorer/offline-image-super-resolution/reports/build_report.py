"""Build the Generative AI mini-project report as a navigable PDF."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "Generative_AI_Mini_Project_Report.pdf"

FONT = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
FONT_I = "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"


class Report(FPDF):
    def __init__(self) -> None:
        super().__init__(format="A4", unit="mm")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 16, 18)
        self.add_font("Serif", "", FONT)
        self.add_font("Serif", "B", FONT_B)
        self.add_font("Serif", "I", FONT_I)
        self.chrome = False
        self.sections: list[tuple[str, int, int]] = []

    def header(self) -> None:
        if not self.chrome:
            return
        self.set_font("Serif", "I", 9)
        self.set_text_color(70, 70, 70)
        self.cell(95, 6, "Generative AI Mini Project", align="L")
        self.cell(0, 6, "Offline AI Image Enhancer", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(40, 40, 40)
        self.line(18, 23, 192, 23)
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        if not self.chrome:
            return
        self.set_y(-14)
        self.set_font("Serif", "", 10)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, str(self.page_no()), align="C")
        self.set_text_color(0, 0, 0)

    def start_body(self) -> None:
        self.chrome = True

    def chapter(self, number: str, title: str) -> None:
        self.add_page()
        label = f"{number}   {title}" if number else title
        self.sections.append((label, self.page_no(), 0))
        self.start_section(label, level=0)
        self.set_font("Serif", "B", 16)
        self.multi_cell(0, 8, label)
        self.ln(2)
        self.set_draw_color(30, 30, 30)
        self.line(18, self.get_y(), 192, self.get_y())
        self.ln(4)

    def h2(self, text: str) -> None:
        self.ln(1)
        self.set_font("Serif", "B", 13)
        self.multi_cell(0, 7, text)
        self.ln(1)
        self.start_section(text, level=1)

    def body(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Serif", "", 12)
        self.multi_cell(0, 6.4, text)
        self.ln(2)

    def bullet(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Serif", "", 12)
        self.multi_cell(0, 6.4, "-  " + text)

    def figure(self, path: Path, caption: str, max_h: float = 145) -> None:
        image = Image.open(path)
        width = self.epw
        height = width * image.height / image.width
        if height > max_h:
            height = max_h
            width = height * image.width / image.height
        if self.get_y() + height + 18 > self.h - self.b_margin:
            self.add_page()
        x = self.l_margin + (self.epw - width) / 2
        self.image(str(path), x=x, w=width, h=height)
        self.ln(1.5)
        self.set_font("Serif", "I", 11)
        self.multi_cell(0, 5.5, caption, align="C")
        self.ln(3)

    def table(self, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
        self.set_font("Serif", "B", 11)
        self.set_fill_color(230, 233, 238)
        for header, width in zip(headers, widths):
            self.cell(width, 8, header, border=1, fill=True)
        self.ln()
        self.set_font("Serif", "", 10.5)
        for row in rows:
            y0 = self.get_y()
            heights = []
            # measure
            lines_needed = 1
            for cell, width in zip(row, widths):
                lines = self.multi_cell(width, 5.5, cell, dry_run=True, output="LINES")
                lines_needed = max(lines_needed, len(lines))
            row_h = max(8, lines_needed * 5.5 + 1.5)
            if y0 + row_h > self.h - self.b_margin:
                self.add_page()
                y0 = self.get_y()
            x0 = self.l_margin
            for cell, width in zip(row, widths):
                self.set_xy(x0, y0)
                self.rect(x0, y0, width, row_h)
                self.multi_cell(width, 5.5, cell)
                x0 += width
            self.set_xy(self.l_margin, y0 + row_h)
        self.ln(3)


def write_cover(pdf: Report) -> None:
    pdf.chrome = False
    pdf.add_page()
    pdf.set_y(28)
    pdf.set_font("Serif", "B", 14)
    pdf.cell(0, 8, "GENERATIVE AI", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Serif", "B", 18)
    pdf.cell(0, 10, "MINI PROJECT REPORT", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_draw_color(20, 20, 20)
    pdf.line(40, pdf.get_y(), 170, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Serif", "B", 16)
    pdf.multi_cell(0, 8, "Offline AI Image Enhancer", align="C")
    pdf.ln(1)
    pdf.set_font("Serif", "", 13)
    pdf.multi_cell(
        0,
        7,
        "Local image super-resolution with Real-ESRGAN\non Apple Silicon",
        align="C",
    )
    pdf.ln(8)
    pdf.line(40, pdf.get_y(), 170, pdf.get_y())
    pdf.ln(12)
    pdf.set_font("Serif", "", 12)
    pdf.cell(0, 7, "Subject", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Serif", "B", 13)
    pdf.cell(0, 8, "Generative AI Mini Project", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Serif", "B", 13)
    pdf.cell(0, 8, "Submitted by", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Serif", "", 12)
    pdf.table_placeholder = None
    # member table centered
    rows = [
        ("Chaitany Kakde", "BT24S05F007"),
        ("Vishal Bhutekar", "BT24S05F002"),
    ]
    pdf.set_font("Serif", "B", 12)
    x = 45
    pdf.set_x(x)
    pdf.set_fill_color(230, 233, 238)
    pdf.cell(70, 8, "Name", border=1, fill=True, align="C")
    pdf.cell(50, 8, "Roll number", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Serif", "", 12)
    for name, roll in rows:
        pdf.set_x(x)
        pdf.cell(70, 8, name, border=1, align="C")
        pdf.cell(50, 8, roll, border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(16)
    pdf.set_font("Serif", "", 12)
    pdf.cell(0, 7, "Academic session 2026–27", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "31 October 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(18)
    pdf.set_font("Serif", "I", 11)
    pdf.multi_cell(
        0,
        6,
        "This report describes the system that was built and run on an Apple MacBook Air M5. "
        "Measurements and screenshots are taken from that machine. Results that were not measured are not stated.",
        align="C",
    )


def write_front(pdf: Report) -> None:
    pdf.start_body()
    pdf.chapter("", "Declaration")
    pdf.body(
        "We, Chaitany Kakde (BT24S05F007) and Vishal Bhutekar (BT24S05F002), declare that this mini project, "
        "Offline AI Image Enhancer, is our own work for the subject Generative AI. The application runs "
        "pretrained Real-ESRGAN weights locally. It does not call a paid inference API. The timings, "
        "checksum, and interface screenshots in this report were recorded on the development Mac on 31 October 2026."
    )
    pdf.ln(8)
    pdf.body("Chaitany Kakde                                          Vishal Bhutekar")
    pdf.body("BT24S05F007                                              BT24S05F002")

    pdf.chapter("", "Acknowledgement")
    pdf.body(
        "We thank the authors of Real-ESRGAN and BasicSR for releasing the model architecture and the "
        "pretrained weights under the BSD 3-Clause license. The project uses those weights unchanged. "
        "PyTorch, Streamlit, Pillow, and scikit-image made a local Apple Silicon implementation practical "
        "without a CUDA GPU."
    )

    pdf.chapter("", "Abstract")
    pdf.body(
        "The project is a local image super-resolution application. A user supplies a low-resolution "
        "photograph or test image. The application reconstructs a larger image with the official "
        "Real-ESRGAN x4plus network and shows the original and the result side by side. Inference runs "
        "on the Mac, using Apple Metal through PyTorch MPS when that device accepts the model, and the "
        "CPU when it does not."
    )
    pdf.body(
        "On the test machine, an Apple MacBook Air with an M5 chip and 16 GB of unified memory, a "
        "320 × 240 built-in sample was enlarged to 1280 × 960. The interface reported 1.15 seconds for "
        "that native 4× run on Apple Metal, using the whole image rather than tiles. A separate timing "
        "of the same resolution measured 1.335 seconds inside the inference call. The official weight "
        "file RealESRGAN_x4plus.pth was checked against SHA-256 "
        "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1."
    )
    pdf.body(
        "The report also records the limits of the build. Face enhancement with GFPGAN is not active, "
        "because those weights were not installed. The native 2× Real-ESRGAN model and the anime model "
        "were not downloaded. Selecting 2× with the x4plus network is a Lanczos resize after a native 4× "
        "pass, and the interface says so. Peak-signal-to-noise ratio and structural similarity are "
        "implemented only for a same-size reference image. They are not shown for an ordinary upload."
    )


def write_chapters(pdf: Report) -> None:
    pdf.chapter("1", "Introduction")
    pdf.h2("1.1  Background")
    pdf.body(
        "Photographs are often kept at a resolution that is too small for a print, a crop, or a display. "
        "Classical resizing, such as bicubic or Lanczos interpolation, can enlarge the pixel grid, but it "
        "cannot recover texture that was never stored. Single-image super-resolution tries to estimate a "
        "plausible high-resolution image from one low-resolution input. Generative adversarial networks "
        "made that estimate sharper than pure regression models, at the cost of sometimes inventing detail."
    )
    pdf.body(
        "Real-ESRGAN is a practical member of that family. It was trained to restore real photographs "
        "that contain blur, noise, and compression, not only clean bicubic downsampling. This mini project "
        "does not train a new network. It packages the published x4plus model in an offline desktop-style "
        "application so a user can enhance an image without sending it to a cloud service."
    )
    pdf.h2("1.2  Need for a local system")
    pdf.body(
        "Cloud enhancers are convenient, but they require a network, an account, and trust that the "
        "photograph can leave the machine. The target computer is a MacBook Air M5. It has no NVIDIA GPU "
        "and therefore no CUDA. A useful project on this hardware has to detect Apple Metal, run the "
        "network there when the operations are supported, and fall back to the CPU without pretending "
        "that the GPU was used."
    )
    pdf.h2("1.3  Scope")
    pdf.bullet("Accept JPEG, PNG, and WEBP images from the local disk.")
    pdf.bullet("Run Real-ESRGAN x4plus from a local weight file for a native 4× enlargement.")
    pdf.bullet("Offer 2× only as a documented resize of the native 4× result, unless a native 2× model is installed.")
    pdf.bullet("Show the original and the enhanced image, then save PNG and JPEG copies.")
    pdf.bullet("Keep inference on the Mac. The running application does not download weights or call an AI API.")
    pdf.body(
        "Training, a full face-restoration pipeline, and a benchmark across many photographs are outside "
        "the delivered scope. Those items are listed as future work rather than as completed results."
    )

    pdf.chapter("2", "Objectives")
    pdf.body("The project set out to do the following, and the report judges each one against what actually runs.")
    pdf.bullet("Build an interactive local application for image super-resolution.")
    pdf.bullet("Use the official Real-ESRGAN x4plus weights, not a substitute or a placeholder filter.")
    pdf.bullet("Run on Apple Silicon, with Metal when it works and the CPU when it does not.")
    pdf.bullet("Show the device that was really selected.")
    pdf.bullet("Let the user compare the input and the output and download the result.")
    pdf.bullet("Refuse unsupported or damaged files with a clear message.")
    pdf.bullet("Avoid reporting PSNR or SSIM when the user has not provided a reference image.")

    pdf.chapter("3", "Literature review")
    pdf.h2("3.1  From interpolation to learned super-resolution")
    pdf.body(
        "Early enlargement methods reconstruct a continuous signal from discrete samples and then resample "
        "it. They are stable and fast. They also smooth edges, because they have no source of new "
        "high-frequency texture. Example-based and sparse-coding methods later searched a dictionary of "
        "patches. Convolutional networks replaced that search with a learned mapping from a low-resolution "
        "tensor to a high-resolution tensor."
    )
    pdf.h2("3.2  SRGAN and ESRGAN")
    pdf.body(
        "SRGAN showed that a generator trained with an adversarial loss and a perceptual loss produces "
        "images that people prefer to those trained with pixel loss alone, even when PSNR is lower. "
        "ESRGAN replaced the residual blocks with residual-in-residual dense blocks (RRDB), removed batch "
        "normalization from the generator, and used a relativistic discriminator. Those changes reduced "
        "some of the artefacts of SRGAN and became the backbone of later real-world models."
    )
    pdf.h2("3.3  Real-ESRGAN")
    pdf.body(
        "Real-ESRGAN, introduced by Wang, Xie, Dong and Shan, extends ESRGAN to unknown real degradations. "
        "A high-order degradation process mixes blur, noise, resizing, and JPEG compression so the "
        "generator sees inputs closer to photographs from cameras and the web. The x4plus checkpoint used "
        "here is the general-photo model: 23 RRDB blocks, 64 feature channels, and a native scale of 4. "
        "The network expects a BGR tensor scaled to the range 0 to 1, which matches the official inference "
        "code. The project follows that convention and converts back to RGB only when the image is shown "
        "or saved."
    )
    pdf.h2("3.4  What this project does not claim")
    pdf.body(
        "A GAN super-resolver can add texture that was not in the input. That texture can look convincing "
        "and still be factually wrong, for example on text, faces, or medical detail. This report treats "
        "the output as a restoration for viewing, not as evidence. GFPGAN is the usual companion model "
        "for faces. It is discussed because it was part of the original design, and it is marked as not "
        "active in the build under test."
    )

    pdf.chapter("4", "System analysis and requirements")
    pdf.h2("4.1  Hardware used for development and testing")
    pdf.table(
        ["Item", "Observed value"],
        [
            ["Computer", "MacBook Air, Apple M5"],
            ["CPU", "10 cores (4 performance, 6 efficiency)"],
            ["Memory", "16 GB unified memory"],
            ["Architecture", "ARM64"],
            ["Operating system", "macOS 27.0.1"],
            ["GPU", "Apple Metal. CUDA is not present and is not used."],
        ],
        [55, 119],
    )
    pdf.body(
        "Sixteen gigabytes of unified memory is shared by macOS and the model. A 4× enlargement of a very "
        "large photograph would allocate a large output tensor. The application therefore limits the "
        "longest input side to 2048 pixels on this class of machine and splits bigger working images into "
        "tiles. The pictured test did not need tiles."
    )
    pdf.h2("4.2  Software requirements")
    pdf.table(
        ["Component", "Version used", "Role"],
        [
            ["Python", "3.13", "Project runtime in the virtual environment"],
            ["PyTorch", "2.15.0", "Network execution, including MPS"],
            ["Torchvision", "0.29.2", "Installed with the PyTorch stack and checked at setup"],
            ["Streamlit", "1.66.1", "Local web interface on 127.0.0.1"],
            ["Pillow", "13.0.0", "Image decode, resize, and save"],
            ["NumPy", "2.5.4", "Tensor layout between Pillow and PyTorch"],
            ["scikit-image", "0.26.0", "PSNR and SSIM when a reference exists"],
        ],
        [40, 32, 102],
    )
    pdf.h2("4.3  Functional requirements")
    pdf.bullet("The user can upload an image or load the built-in sample.")
    pdf.bullet("The user can select a model that is actually present on disk. Missing models are listed as not downloaded.")
    pdf.bullet("The user can choose 2× or 4×. The screen states when 2× is not a native network.")
    pdf.bullet("The user can start processing and see the device, the time, and the output size.")
    pdf.bullet("The user can compare the images and download PNG and JPEG.")
    pdf.bullet("Several images can be queued and processed one after another.")
    pdf.h2("4.4  Non-functional requirements")
    pdf.bullet("Inference must not depend on the network once the weights are on disk.")
    pdf.bullet("The displayed device must match the device that ran the model.")
    pdf.bullet("A damaged file must fail with a message, not with a traceback shown as a result.")
    pdf.bullet("Uploaded bytes are checked before they are decoded. The application does not execute the file.")

    pdf.chapter("5", "System design")
    pdf.h2("5.1  Architecture")
    pdf.body(
        "The design separates the interface, the checks, the network, and the device choice. Streamlit "
        "only collects a request and displays a result. It does not contain the Real-ESRGAN arithmetic. "
        "That split makes the same engine usable from the button in the browser and from a short Python "
        "timing script."
    )
    pdf.figure(FIG / "fig_architecture.png", "Figure 1. Runtime architecture. Every stage reads and writes local data.")
    pdf.h2("5.2  Processing sequence")
    pdf.bullet("The interface reads the file bytes and records the original size.")
    pdf.bullet("Validation checks the extension, the file signature, the decoder, and the pixel limits.")
    pdf.bullet("The image is converted to RGB. An alpha channel is composited onto white.")
    pdf.bullet("The device manager probes MPS with a real matrix multiply. If the probe fails, the choice is CPU.")
    pdf.bullet("The model manager loads the state dict from models/ and checks that the keys match RRDBNet.")
    pdf.bullet("The inference engine converts RGB to the BGR float tensor used by Real-ESRGAN.")
    pdf.bullet("The network runs, with tiles if the image is large. The tensor is clamped, converted back to RGB, and saved.")
    pdf.h2("5.3  Scale policy")
    pdf.body(
        "Real-ESRGAN x4plus has one native scale: 4. The project does not describe a 2× selection as a "
        "second forward pass of a 2× network. If the user asks for 2×, the engine still runs the 4× "
        "network and then resizes the result to twice the input with Lanczos resampling. The result panel "
        "states that strategy. A true native 2× model, RealESRGAN_x2plus, is defined in the catalog and "
        "will be offered if its weight file is added later. It was not on disk for this report."
    )
    pdf.h2("5.4  Memory policy")
    pdf.body(
        "Small images, including the 320 × 240 sample, are forwarded in one piece. Larger images use "
        "overlapping tiles of 128 to 512 pixels, following the tiling scheme of the official Real-ESRGAN "
        "tester, with a 10-pixel pad so seams are less visible. If a run raises a memory error, the engine "
        "retries a smaller tile and then the CPU. The result records the device that succeeded."
    )

    pdf.chapter("6", "Implementation")
    pdf.h2("6.1  Project layout")
    pdf.body(
        "The working tree is offline-image-super-resolution. app.py is the Streamlit entry. config/settings.py "
        "holds paths, the model catalog, and checksums. core/ holds validation, the RRDB and compact VGG "
        "architectures, model loading, inference, metrics, and device selection. ui/ holds the sidebar, "
        "the comparison panel, and the dark theme. models/ holds weight files. output/ holds saved results. "
        "run.sh activates the virtual environment, checks that at least one upscaler is present, and starts "
        "Streamlit on 127.0.0.1 port 8501."
    )
    pdf.h2("6.2  Network implementation")
    pdf.body(
        "The RRDB generator is implemented locally so the project does not depend on the BasicSR package, "
        "which has been fragile on new PyTorch releases. Layer names match the official checkpoint: "
        "conv_first, 23 RRDB body blocks, conv_body, two nearest-neighbour upsample stages, and conv_last. "
        "Loading uses strict key matching. A mismatch is reported as a weight error instead of a silent "
        "partial load. The general x4v3 model, which is also present, uses the compact SRVGGNet with "
        "32 convolutions and a pixel-shuffle upsample. It is a faster restoration network, not the primary "
        "model of this report."
    )
    pdf.h2("6.3  Interface")
    pdf.body(
        "The screen is a dark workspace named OFFLINE AI IMAGE ENHANCER. The top row shows that inference "
        "is local, which device will be used, how many images have been processed, and the accumulated "
        "time. Those counters are stored in output/processing_stats.json. The sidebar lists every catalog "
        "model as Ready or Not downloaded, then offers the device, the ready model, the scale, the tile "
        "size, and the JPEG quality. The Enhance tab accepts a file or the built-in sample. The Batch tab "
        "accepts several files and processes them in order so two large networks are not resident together "
        "beyond the single loaded model."
    )
    pdf.figure(
        FIG / "ui_home.png",
        "Figure 2. Application home on the test Mac. Real-ESRGAN x4plus is selected, 4× is selected, and Metal is the resolved device.",
    )
    pdf.h2("6.4  Colour and file safety")
    pdf.body(
        "JPEG, PNG, and WEBP are accepted only when the extension and the leading bytes agree and Pillow "
        "can decode the file. A renamed or truncated file is rejected. The application limit is 25 MB and "
        "a longest side of 2048 pixels, even though the Streamlit upload control itself prints a larger "
        "framework limit. Outputs are written with sanitized file names into output/. Paths shown to the "
        "user are file names, not a tour of the home directory."
    )
    pdf.h2("6.5  Metrics module")
    pdf.body(
        "core/image_metrics.py computes PSNR and SSIM with scikit-image. Both functions refuse arrays of "
        "different shapes, and SSIM also refuses an image smaller than the comparison window. Identical "
        "images produce an infinite PSNR and an SSIM of 1. The interface does not call this module for a "
        "normal upload. Doing so would compare the enhancement with the low-resolution input after an "
        "arbitrary resize, which is not a ground-truth score."
    )

    pdf.chapter("7", "Results and discussion")
    pdf.h2("7.1  Weight integrity")
    pdf.body(
        "RealESRGAN_x4plus.pth was downloaded from the official Real-ESRGAN v0.1.0 release. Its size is "
        "67,040,989 bytes. Its SHA-256 digest matches the published digest "
        "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1. The compact general model "
        "realesr-general-x4v3.pth is also present and matches digest "
        "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292. Incomplete downloads of the "
        "anime model, the native 2× model, and GFPGAN were removed so they cannot be loaded by mistake."
    )
    pdf.h2("7.2  A measured 4× run")
    pdf.body(
        "The built-in sample is a 320 × 240 RGB drawing used so the application can be tried without a "
        "photograph. It is not a camera image, and it has no high-resolution ground truth. It does test "
        "the full path: decode, 4× inference, display, and save."
    )
    pdf.table(
        ["Quantity", "Measured value"],
        [
            ["Model", "Real-ESRGAN x4plus, native 4×"],
            ["Device reported by the application", "Apple Metal (MPS)"],
            ["Tile size", "Whole image"],
            ["Input", "320 × 240"],
            ["Output", "1280 × 960"],
            ["Time shown in the interface", "1.15 s"],
            ["Inference time in a repeat script run", "1.335 s"],
            ["Saved PNG", "1.12 MB"],
            ["Strategy text", "Native 4× inference."],
        ],
        [78, 96],
    )
    pdf.body(
        "The two times are not a contradiction. The interface time is the synchronized model run for the "
        "interactive session. The script time is a later call after the weights were loaded again in "
        "another process. Both finished on MPS, and both produced 1280 × 960. An earlier smoke test of a "
        "96 × 64 input produced 384 × 256 in 0.441 seconds on the same device."
    )
    pdf.figure(
        FIG / "fig_before_after.png",
        "Figure 3. Built-in sample and the 4× Real-ESRGAN result. The right-hand image is shown reduced so both fit on the page; the file is 1280 × 960.",
        max_h=95,
    )
    pdf.figure(
        FIG / "ui_session.png",
        "Figure 4. Result screen: original preview, enhanced preview, and the comparison control.",
        max_h=150,
    )
    pdf.h2("7.3  Reading the pictures")
    pdf.body(
        "The input drawing contains flat colour, a hard white rectangle, and a thin diagonal line. The "
        "4× output keeps those structures and the frame size is exactly four times the input on each side, "
        "which is the check that the scale factor was not merely a label. Fine stair-stepping on the "
        "diagonal is still visible. That is expected: the sample was drawn at 320 × 240 with hard pixel "
        "edges, and a photographic restoration model is not a vector tracer. The useful engineering result "
        "is that colour order stayed consistent (the blue field remains blue and the gold disc remains gold) "
        "and that the official network executed on Metal."
    )
    pdf.body(
        "This sample should not be used to claim a PSNR. There is no original high-resolution photograph "
        "from which the 320 × 240 image was downsampled. Publishing a score against the input, or against "
        "a resized copy of the output, would be a made-up evaluation. The metrics code remains available "
        "for a later test in which a known reference and a controlled downsample share the same dimensions."
    )
    pdf.h2("7.4  Interface behaviour that was observed")
    pdf.bullet("The home screen shows Local inference and Apple Metal (MPS) when Auto is selected.")
    pdf.bullet("x4plus is marked Ready. x2plus and the anime model are marked Not downloaded.")
    pdf.bullet("After enhancement, the panel shows 1280 × 960, 1.15 s, 4×, and whole-image tiling.")
    pdf.bullet("PNG and JPEG download controls are present, and both files are written under output/.")
    pdf.bullet("The persistent counter on a later page load showed 3 processed images and 3.6 seconds total.")
    pdf.bullet("The face-enhancement note states that the control is off until GFPGAN weights exist.")

    pdf.chapter("8", "Testing")
    pdf.h2("8.1  Tests that were executed")
    pdf.table(
        ["Test", "Result"],
        [
            ["Python 3.13 virtual environment", "Created and used for every run"],
            ["Import of PyTorch 2.15", "Passed. MPS reported available"],
            ["MPS numeric probe", "A 4×4 matrix multiply on MPS returned to CPU"],
            ["x4plus checksum", "Matched the official SHA-256"],
            ["x4plus state-dict load", "Keys matched the local RRDBNet"],
            ["96 × 64 to 384 × 256", "Passed on MPS in 0.441 s"],
            ["320 × 240 to 1280 × 960", "Passed on MPS; interface time 1.15 s"],
            ["PNG and JPEG reopen", "Both saved files open as RGB at 1280 × 960"],
            ["Streamlit process", "Served http://127.0.0.1:8501 and rendered the dashboard"],
            ["Sample button and Start enhancement", "Produced the comparison and download controls"],
        ],
        [78, 96],
    )
    pdf.h2("8.2  What was not claimed as a passed test")
    pdf.body(
        "A full pytest suite, a three-resolution CPU-versus-MPS benchmark table, and an offline firewall "
        "trial were not completed for this shorter delivery. The report therefore does not include a "
        "benchmark table. CPU fallback is implemented and can be selected in the sidebar. It was not timed "
        "as a separate published experiment. GFPGAN was not tested, because the weight file is absent."
    )
    pdf.h2("8.3  Judgement")
    pdf.body(
        "The project meets the core brief of a Generative AI mini project: a real pretrained generative "
        "restoration model, running locally, with an interface a user can operate. The evidence is the "
        "checksum, the Metal timing, the exact 4× geometry, and the screenshots. The evidence is not a "
        "leaderboard score. Photographic quality should be judged by enhancing a picture the user cares "
        "about, and any numeric quality claim should wait for a paired reference."
    )

    pdf.chapter("9", "Conclusion and future scope")
    pdf.h2("9.1  Conclusion")
    pdf.body(
        "Offline AI Image Enhancer loads official Real-ESRGAN x4plus weights and enlarges an image on an "
        "Apple M5 MacBook without a cloud model. The tested path takes a 320 × 240 input to a 1280 × 960 "
        "PNG and JPEG, reports Apple Metal as the device, and keeps the files on the machine. The "
        "interface distinguishes a native 4× run from a 2× resize, and it refuses to invent PSNR and SSIM."
    )
    pdf.h2("9.2  Future scope")
    pdf.bullet("Download and enable RealESRGAN_x2plus so 2× is a native network rather than a resize.")
    pdf.bullet("Add the anime 6-block model for illustrations.")
    pdf.bullet("Install GFPGANv1.4 and a face detector, then enable the face control that is already described in the sidebar.")
    pdf.bullet("Add a reference-image control that computes PSNR and SSIM only when the sizes match.")
    pdf.bullet("Publish a CPU and MPS timing table for 256, 512, and 1024 inputs after those runs are measured.")
    pdf.bullet("Pair freely licensed photographs with controlled downsamples so visual results are photographic as well as geometric.")

    pdf.chapter("", "References")
    pdf.bullet("C. Ledig et al., “Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network,” CVPR, 2017.")
    pdf.bullet("X. Wang et al., “ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks,” ECCV Workshops, 2018.")
    pdf.bullet("X. Wang, L. Xie, C. Dong, and Y. Shan, “Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data,” ICCV Workshops, 2021.")
    pdf.bullet("X. Wang, Y. Li, H. Zhang, and Y. Shan, “Towards Real-World Blind Face Restoration with Generative Facial Prior,” CVPR, 2021. Discussed as related work. Not executed in this build.")
    pdf.bullet("Xintao Wang et al., Real-ESRGAN source and v0.1.0 weights, BSD 3-Clause license, https://github.com/xinntao/Real-ESRGAN")
    pdf.bullet("PyTorch documentation, MPS backend, version 2.15.")
    pdf.bullet("Streamlit documentation, version 1.66.")

    pdf.chapter("", "Appendix A. How to run the project")
    pdf.body("From the project directory:")
    pdf.body('cd "offline-image-super-resolution"')
    pdf.body("./run.sh")
    pdf.body("Then open http://127.0.0.1:8501 . Choose Use built-in sample, leave 4× selected, and press Start enhancement.")
    pdf.body(
        "The first-time setup, already completed on the test Mac, is a Python 3.13 virtual environment, "
        "pip install -r requirements.txt, and python scripts/download_models.py for any missing official weights."
    )


def _prepare(pdf: Report) -> None:
    pdf.set_title("Offline AI Image Enhancer — Generative AI Mini Project")
    pdf.set_author("Chaitany Kakde (BT24S05F007); Vishal Bhutekar (BT24S05F002)")
    pdf.set_subject("Generative AI Mini Project")
    pdf.set_creator("Offline AI Image Enhancer")


def write_contents(pdf: Report, entries: list[tuple[str, int]]) -> None:
    pdf.chapter("", "Contents")
    pdf.set_font("Serif", "", 12)
    for label, page in entries:
        if label == "Contents":
            continue
        pdf.set_x(pdf.l_margin)
        y = pdf.get_y()
        pdf.cell(150, 7, label)
        pdf.cell(24, 7, str(page), align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.set_xy(pdf.l_margin, y)
        pdf.ln(7)


def main() -> None:
    draft = Report()
    _prepare(draft)
    write_cover(draft)
    write_front(draft)
    write_chapters(draft)
    abstract_page = next(page for label, page, _level in draft.sections if label == "Abstract")
    shifted = []
    for label, page, _level in draft.sections:
        shown = page if page <= abstract_page else page + 1
        shifted.append((label, shown))

    pdf = Report()
    _prepare(pdf)
    write_cover(pdf)
    write_front(pdf)
    write_contents(pdf, shifted)
    write_chapters(pdf)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print(f"pages={pdf.page} file={OUT} bytes={OUT.stat().st_size}")
    for label, page, _level in pdf.sections:
        print(f"{page:3}  {label}")


if __name__ == "__main__":
    main()
