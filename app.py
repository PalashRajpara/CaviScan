import streamlit as st
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from PIL import Image
import uuid
import os
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage

# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="Your_API_Key_Here"
)

# Streamlit app configuration
st.set_page_config(page_title="CaviScan", layout="wide")
st.title("CaviScan 🦷 - Dental Cavity Detection Using X-ray Imaging")
st.markdown("Upload a dental X-ray image to detect cavities.")

# Sidebar for image upload and report generation
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Add report section to sidebar
st.sidebar.markdown("---")
st.sidebar.header("📑 Report Section")

# Function to process and display results
def process_image(image_file):
    # Save the uploaded image temporarily
    temp_image_path = f"temp_image_{uuid.uuid4()}.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(image_file.read())

    # Perform inference
    result = CLIENT.infer(temp_image_path, model_id="cavity1.0/1")

    # Load the image for visualization
    img = cv2.imread(temp_image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process predictions
    detections = result.get("predictions", [])
    detection_summary = []
    
    for detection in detections:
        x = int(detection["x"])
        y = int(detection["y"])
        width = int(detection["width"])
        height = int(detection["height"])
        class_name = detection["class"]
        confidence = detection["confidence"]

        # Calculate bounding box coordinates
        x0 = int(x - width / 2)
        x1 = int(x + width / 2)
        y0 = int(y - height / 2)
        y1 = int(y + height / 2)

        # Draw bounding box
        cv2.rectangle(img_rgb, (x0, y0), (x1, y1), (255, 0, 0), 2)
        # Add text label with confidence percentage
        label = f"{class_name}: {confidence*100:.1f}%"
        cv2.putText(
            img_rgb,
            label,
            (x0, y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )
        # Collect detection details for text output without box coordinates
        detection_summary.append(f"Class: {class_name}, Confidence: {confidence*100:.1f}%")

    # Save processed image for report
    processed_image_path = f"temp_image_processed_{uuid.uuid4()}.jpg"
    cv2.imwrite(processed_image_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # Clean up temporary file
    os.remove(temp_image_path)

    return img_rgb, detection_summary, processed_image_path, detections

# Function to generate PDF report
def generate_report(predictions, processed_image_path):
    # Create a BytesIO object to store the PDF
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Create a custom title style
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=24,
        alignment=1,  # Center alignment
        spaceAfter=12
    )
    
    # Add title
    title = Paragraph("CaviScan Dental Report", title_style)
    elements.append(title)
    elements.append(Spacer(1, 20))
    
    # Add date
    date_style = ParagraphStyle(
        'Date',
        parent=styles['Normal'],
        fontSize=12,
        alignment=1,  # Center alignment
        spaceAfter=12
    )
    current_date = datetime.now().strftime("%B %d, %Y")
    date_text = Paragraph(f"Report generated on: {current_date}", date_style)
    elements.append(date_text)
    elements.append(Spacer(1, 20))
    
    # Add summary
    summary_style = ParagraphStyle(
        'Summary',
        parent=styles['Normal'],
        fontSize=14,
        alignment=0,  # Left alignment
        spaceAfter=12
    )
    
    if not predictions:
        summary_text = Paragraph("No cavities detected.", summary_style)
    else:
        summary_text = Paragraph(f"Cavities detected: {len(predictions)}", summary_style)
    elements.append(summary_text)
    elements.append(Spacer(1, 10))
    
    # Add details if cavities are detected
    if predictions:
        detail_style = ParagraphStyle(
            'Detail',
            parent=styles['Normal'],
            fontSize=12,
            leftIndent=20,
            spaceAfter=6
        )
        elements.append(Paragraph("Detection Details:", styles['Heading2']))
        for i, prediction in enumerate(predictions, 1):
            confidence = prediction["confidence"] * 100
            elements.append(
                Paragraph(
                    f"{i}. Class: {prediction['class']}, Confidence: {confidence:.1f}%",
                    detail_style
                )
            )
        elements.append(Spacer(1, 20))
    
    # Add image if available
    if os.path.exists(processed_image_path):
        try:
            img = Image.open(processed_image_path)
            width, height = img.size
            aspect = width / height
            
            # Scale image to fit on page
            max_width = 450
            img_width = min(max_width, width)
            img_height = img_width / aspect
            
            # Add image to PDF
            img_for_report = ReportLabImage(processed_image_path, width=img_width, height=img_height)
            elements.append(Paragraph("Processed X-ray Image:", styles['Heading2']))
            elements.append(img_for_report)
        except Exception as e:
            elements.append(Paragraph("⚠️ Error loading processed image", styles['Normal']))
    else:
        elements.append(Paragraph("⚠️ Processed image not available", styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    
    # Get the value from the BytesIO buffer
    buffer.seek(0)
    return buffer

# Main app logic
if uploaded_file is not None:
    # Display the uploaded image and results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(uploaded_file, caption="Original Image", use_container_width=True)
    
    with col2:
        st.subheader("Detection Results")
        try:
            # Process the image and get results
            processed_img, detection_summary, processed_image_path, raw_detections = process_image(uploaded_file)
            # Display the processed image with bounding boxes
            st.image(processed_img, caption="Detected Cavities", use_container_width=True)
            # Display detection details as text
            if detection_summary:
                st.write("**Detection Results:**")
                for detail in detection_summary:
                    st.write(detail)
            else:
                st.write("No cavities detected.")
                
            # Report generation section in sidebar
            if st.sidebar.button("📄 Generate Report"):
                # Generate PDF report
                pdf_buffer = generate_report(raw_detections, processed_image_path)
                
                # Create download button
                st.sidebar.download_button(
                    label="⬇️ Download Report as PDF",
                    data=pdf_buffer,
                    file_name="CaviScan_Report.pdf",
                    mime="application/pdf"
                )
                st.sidebar.success("Report generated successfully!")
                
            # Clean up processed image file when done
            if os.path.exists(processed_image_path):
                try:
                    os.remove(processed_image_path)
                except:
                    pass
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
else:
    st.write("Please upload an image to start detection.")
    st.sidebar.info("Upload an image to generate a report.")

