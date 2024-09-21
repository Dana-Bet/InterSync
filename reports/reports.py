import os
import pandas as pd
from pandas import DataFrame


from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, ListItem, ListFlowable, PageBreak

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

KEYFRAME_PLOT_FILENAME = 'frame_pixel_difference.png'
SIMILARITY_PLOT_FOLDER_NAME = 'similarity'
DTW_PLOT_FOLDER_NAME = 'dtw'
SMITH_WATERMAN_PLOT_FOLDER_NAME = 'smith_waterman'
TLCC_PLOT_FOLDER_NAME = 'tlcc'

# Custom styles for the report
title_style = ParagraphStyle(
    name='TitleStyle',
    fontSize=18,
    leading=22,
    alignment=1,  # Center align
    spaceAfter=10
)
subtitle_style = ParagraphStyle(
    name='SubtitleStyle',
    fontSize=12,
    leading=14,
    alignment=1,  # Center align
    spaceAfter=20
)
section_title_style = ParagraphStyle(
    name='SectionTitleStyle',
    fontSize=12,
    leading=14,
    spaceBefore=20,
    spaceAfter=5,
    fontName="Helvetica-Bold"
)

normal_centered_style = ParagraphStyle(
    name='NormalCentered',
    fontSize=10,
    leading=12,
    alignment=1
)


def create_table_from_dataframe(dataframe):
    dataframe = dataframe.astype(str)
    data = [dataframe.columns.tolist()] + dataframe.values.tolist()
    table = Table(data)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.gray),  # header background color
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),  # header text color
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # center alignment
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # header font
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),  # header padding
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),  # cell background color
        ('GRID', (0, 0), (-1, -1), 1, colors.black),  # grid lines
    ])
    table.setStyle(style)
    return table


def attach_images_from_folder(folder_path, num_of_img_per_page=1, padding_factor=0.95):
    c = []
    images_on_page = 0  # Counter for images on the current page

    # Dimensions of the page
    page_width, page_height = A4

    # Calculate available height for each image based on the number of images per page
    available_height_per_image = (page_height / num_of_img_per_page) * padding_factor  # Apply padding factor
    available_width_per_image = page_width * padding_factor  # Apply padding factor for width

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img = Image(file_path)

            # Get the image's original dimensions
            img_width, img_height = img.wrap(0, 0)

            # Scale the image to fit within the available width and height per image
            scaling_factor_width = available_width_per_image / img_width
            scaling_factor_height = available_height_per_image / img_height
            scaling_factor = min(scaling_factor_width, scaling_factor_height)

            img_width = img_width * scaling_factor
            img_height = img_height * scaling_factor

            # Apply the scaling to the image
            img._restrictSize(img_width, img_height)

            # Add the image to the content list
            c.append(Spacer(1, 0.2 * inch))  # Add some space before the image
            c.append(img)

            images_on_page += 1  # Increment image count on the current page

            # If the number of images on the page reaches the limit, add a page break
            if images_on_page >= num_of_img_per_page:
                c.append(PageBreak())  # Add a page break
                images_on_page = 0  # Reset the counter for the next page
    return c


def add_background_logo_header_footer(canvas_obj, doc, logo_path):
    logo_width = 2 * inch
    logo_height = 1 * inch
    page_width, page_height = A4
    x_position = page_width - logo_width - 0.5 * inch

    canvas_obj.drawImage(logo_path, x=x_position, y=page_height - 1.2 * inch,
                         width=logo_width, height=logo_height,
                         preserveAspectRatio=True, mask='auto')
    canvas_obj.saveState()
    canvas_obj.setFont('Helvetica', 9)
    canvas_obj.drawString(0.5 * inch, page_height - 0.5 * inch, "24-1-R-17")
    page_number_text = f"Page {doc.page}"
    canvas_obj.drawString(page_width - 1.5 * inch, 0.5 * inch, page_number_text)
    canvas_obj.restoreState()


def generate_intersync_report_pdf(path: str, similarity_report_df: DataFrame, dtw_report_df: DataFrame,
                                  smith_waterman_report_df: DataFrame, tlcc_unified_report_df: DataFrame,
                                  tlcc_separate_report_df: DataFrame,
                                  keyframe_page_present: bool,
                                  keyframe_page_threshold: int = 0,
                                  frames_analysed: int = 0,
                                  video_filename: str = 'File Name',
                                  run_timestamp: str = '2024-12-31-24-59-59', explanation_text: bool = True,
                                  attach_plot_images: bool = True):
    pdf_file_name = os.path.join(path, "intersync_summary_report.pdf")
    pdf = SimpleDocTemplate(pdf_file_name, pagesize=A4)
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    content = []
    list_of_analysis = []
    if similarity_report_df is not None:
        list_of_analysis.append('Similarity Analysis')
    if dtw_report_df is not None:
        list_of_analysis.append('Dynamic time warping (DTW) Analysis')
    if dtw_report_df is not None:
        list_of_analysis.append('Smith-Waterman Analysis')
    if dtw_report_df is not None:
        list_of_analysis.append('Time-Lagged Cross-Correlation (TLCC) Analysis')
    cover_page = generate_cover_page(normal_style=normal_style, video_filename=video_filename,
                                     run_timestamp=run_timestamp, frames_analysed=frames_analysed,
                                     name1='Semion Rodman', name2='Dana Soudry', advisor_name='Dr. Anat Dahan',
                                     analysis_list=list_of_analysis)
    content.extend(cover_page)
    if keyframe_page_present:
        keyframe_page = generate_keyframe_page(normal_style=normal_style, path=path,
                                               keyframe_detection_threshold=keyframe_page_threshold,
                                               keyframe_number_of_frames=frames_analysed,
                                               explanation_text=explanation_text)
        content.extend(keyframe_page)
    information_page = generate_information_page(normal_style=normal_style, new_page=not keyframe_page_present)
    content.extend(information_page)
    similarity_report_page = generate_similarity_page(normal_style=normal_style, path=path,
                                                      analysis_summary_df=similarity_report_df,
                                                      explanation_text=explanation_text,
                                                      attach_plot_images=attach_plot_images)
    content.extend(similarity_report_page)
    dtw_report_page = generate_dtw_page(normal_style=normal_style, path=path, analysis_summary_df=dtw_report_df,
                                        explanation_text=explanation_text, attach_plot_images=attach_plot_images)
    content.extend(dtw_report_page)
    smith_waterman_report_page = generate_smith_waterman_page(normal_style=normal_style, path=path,
                                                              analysis_summary_df=smith_waterman_report_df,
                                                              explanation_text=explanation_text,
                                                              attach_plot_images=attach_plot_images)
    content.extend(smith_waterman_report_page)
    tlcc_report_page = generate_tlcc_page(normal_style=normal_style, path=path,
                                          unified_analysis_summary_df=tlcc_unified_report_df,
                                          separate_analysis_summary_df=tlcc_separate_report_df,
                                          explanation_text=explanation_text, attach_plot_images=attach_plot_images)
    content.extend(tlcc_report_page)
    pdf.build(content,
              onFirstPage=lambda canvas_obj, doc: add_background_logo_header_footer(canvas_obj, doc,
                                                                                    './images/braude_logo.png'),
              onLaterPages=lambda canvas_obj, doc: add_background_logo_header_footer(canvas_obj, doc,
                                                                                     './images/braude_logo.png'))
    print(f"PDF report '{pdf_file_name}' generated successfully.")


def generate_cover_page(normal_style, video_filename='File Name', run_timestamp='2024-12-31-24-59-59', frames_analysed=0,
                        name1='Name Lastname', name2='Name Lastname', advisor_name='Name Lastname', analysis_list=[]):
    c = []
    c.append(Spacer(1, 0.2 * inch))
    c.append(Paragraph("<u><b>InterSync Run Summary Report</b></u>", title_style))
    c.append(Paragraph("24-1-R-17 Evaluation of Interpersonal Synchronization", subtitle_style))
    c.append(Paragraph("Automated report generated by InterSync", subtitle_style))
    c.append(Spacer(1, 0.2 * inch))
    # analyzed file and timestamp section
    c.append(Paragraph(f"<b>Analyzed file:</b> {video_filename}", normal_style))
    c.append(Paragraph(f"<b>Analysis time:</b> {run_timestamp}", normal_style))
    c.append(Paragraph(f"<b>Analyzed frames:</b> {frames_analysed}", normal_style))
    c.append(Spacer(1, 0.3 * inch))
    # analysis performed section

    c.append(Paragraph("<b>Analysis performed:</b>", section_title_style))
    c.append(Spacer(1, 0.3 * inch))
    if len(analysis_list) != 0:
        bullet_points = ListFlowable(
            [ListItem(Paragraph(item, normal_style)) for item in analysis_list],
            bulletType='bullet')
        c.append(bullet_points)

    c.append(Spacer(1, 1 * inch))
    # 'project by' section
    c.append(Paragraph("<b>Project By:</b>", section_title_style))
    bullet_points = [
        ListItem(Paragraph(name1, normal_style)),
        ListItem(Paragraph(name2, normal_style))
    ]
    bullet_list = ListFlowable(bullet_points, bulletType='bullet', start='•')
    c.append(bullet_list)
    c.append(Spacer(1, 0.5 * inch))
    c.append(Paragraph("<b>Project Advisor:</b>", section_title_style))
    bullet_points = [
        ListItem(Paragraph(advisor_name, normal_style))
    ]
    c.append(ListFlowable(bullet_points, bulletType='bullet', start='•'))
    return c


def generate_information_page(normal_style, new_page=False):
    c = []
    if new_page:
        c.append(PageBreak())
    c.append(Paragraph("General Analysis Information:", section_title_style))
    c.append(Paragraph("The original measurements for this report were obtained using landmark pose detection with "
                       "MediaPipe, where the coordinates were based on the image frame. For the analysis, the vectors "
                       "were transformed into a coordinate system centered on each detection target as follows: the "
                       "X-axis represents the direction from the body center to the head, the Y-axis from the left "
                       "shoulder to the right shoulder, and the Z-axis from the chest forward.", normal_style))
    c.append(Paragraph("Following illustration visualizes the coordinate system used:", normal_style))
    c.append(Spacer(1, 0.3 * inch))
    # try:
    #     with Image('../images/axis_illustration.png') as img:
    #         print("File opened successfully")
    # except FileNotFoundError:
    #     print("File not found")
    # except OSError as e:
    #     print(f"Error: {e}")

    img = Image('./images/axis_illustration.png')

    img_width, img_height = img.wrap(0, 0)
    scaling_factor = 0.5 if new_page else 0.3
    img_width = img_width * scaling_factor
    img_height = img_height * scaling_factor
    img._restrictSize(img_width, img_height)
    c.append(img)
    c.append(Spacer(1, 0.3 * inch))
    return c


def generate_keyframe_page(normal_style, path, keyframe_detection_threshold,
                           keyframe_number_of_frames=0, explanation_text=True):
    keyframe_detection_plot = os.path.join(path, KEYFRAME_PLOT_FILENAME)
    c = [PageBreak()]
    c.append(Paragraph("Keyframes Detection Plot:", section_title_style))
    if explanation_text:
        c.append(Paragraph("Keyframe detection performed by analyzing the frame-to-frame pixel "
                           "differences, identifying significant changes that rise above the threshold value, "
                           "and choosing keyframes for future analysis based on detected peaks", normal_style))

    c.append(Spacer(1, 0.3 * inch))
    img = Image(keyframe_detection_plot)
    img_width, img_height = img.wrap(0, 0)
    scaling_factor = 0.5
    img_width = img_width * scaling_factor
    img_height = img_height * scaling_factor
    img._restrictSize(img_width, img_height)
    c.append(img)
    c.append(Spacer(1, 0.3 * inch))
    c.append(Paragraph(f"Threshold value used: {keyframe_detection_threshold}",
                       normal_centered_style))
    if keyframe_number_of_frames > 0:
        c.append(Paragraph(f"Number of keyframes detected: {keyframe_number_of_frames}",
                           normal_centered_style))
    c.append(Spacer(1, 0.5 * inch))
    return c


def generate_similarity_page(normal_style, path, analysis_summary_df,
                             explanation_text=True, attach_plot_images=True):
    similarity_plots_folder = os.path.join(path, SIMILARITY_PLOT_FOLDER_NAME)
    c = [PageBreak()]
    c.append(Paragraph("<u>Similarity Analysis:</u>", section_title_style))
    if explanation_text:
        c.append(Paragraph("Basic similarity analysis, performed on the extracted movement data, compares body "
                           "part movements "
                           "between two targets by looking at separate segments of movement over time. Each segment "
                           "of first Target is compared against all segments of second Target using chosen similarity "
                           "metric. For each movement segment of first Target, "
                           "we identify similar movement segments of second Target based on the specified similarity "
                           "threshold", normal_style))
        c.append(Spacer(1, 0.2 * inch))
    c.append(Paragraph("Following table presents data gathered during similarity analysis:", normal_style))
    c.append(Spacer(1, 0.2 * inch))
    if explanation_text:
        bullet_points = [
            ListItem(Paragraph("<b>Threshold:</b> maximum allowable distance between vectors for them to be "
                               "considered similar", normal_style)),
            ListItem(Paragraph("<b>Normalized:</b> indicates the data was normalized to remove "
                               "scale differences", normal_style)),
            ListItem(Paragraph("<b>Metric:</b> distance metric used to compare vectors", normal_style)),
            ListItem(Paragraph("<b>Total of Similar Segments:</b> total number of similarities found", normal_style))
        ]
        bullet_list = ListFlowable(bullet_points, bulletType='bullet', start='•')
        c.append(bullet_list)
        c.append(Spacer(1, 0.2 * inch))
    c.append(create_table_from_dataframe(analysis_summary_df))
    if attach_plot_images:
        c.extend(attach_images_from_folder(similarity_plots_folder, 3, padding_factor=0.75))
    return c


def generate_dtw_page(normal_style, path, analysis_summary_df, explanation_text=True, attach_plot_images=True):
    dtw_plots_folder = os.path.join(path, DTW_PLOT_FOLDER_NAME)
    c = [PageBreak()]
    c.append(Paragraph("<u>Dynamic time warping (DTW) Analysis</u>", section_title_style))
    if explanation_text:
        c.append(Paragraph("DTW algorithm is being used to measure the similarity between two "
                           "time-dependent sequences of movement vectors, despite assumed differences in speed or "
                           "length. We are aligning the sequences of vectors by minimizing the cumulative distance "
                           "between them based on a calculation metric chosen."
                           "Additionally, a cost matrix is generated to quantify the individual distances between "
                           "each pair of vectors, to give more insight into the overall alignment",
                           normal_style))
        c.append(Spacer(1, 0.2 * inch))
    c.append(Paragraph("Following table presents data gathered during DTW analysis:", normal_style))
    c.append(Spacer(1, 0.2 * inch))
    if explanation_text:
        bullet_points = [
            ListItem(Paragraph("<b>Metric:</b> distance metric used to compare vectors", normal_style)),
            ListItem(
                Paragraph("<b>Radius:</b> distance within which points from the two sequences are allowed to align",
                          normal_style)),
            ListItem(Paragraph("<b>Normalized:</b> indicates the data was normalized to remove "
                               "scale differences", normal_style)),
            ListItem(Paragraph("<b>Smooth:</b> indicates the sequences were smoothed to reduce noise", normal_style)),
            ListItem(
                Paragraph("<b>Smooth Window Size:</b> specifies the window size used for smoothing", normal_style)),
            ListItem(
                Paragraph("<b>DTW Distance:</b> total distance between sequences representing their overall similarity",
                          normal_style)),
            ListItem(Paragraph(
                "<b>DTW Path Length:</b> length of the optimal alignment - complexity of the sequence alignment",
                normal_style)),
            ListItem(Paragraph(
                "<b>Cost Matrix Shape:</b> dimensions of 'accumulated cost matrix' - number of vectors compared",
                normal_style)),
            ListItem(Paragraph("<b>Min Cost:</b> smallest distance calculated between any two vectors", normal_style)),
            ListItem(Paragraph("<b>Mean Cost:</b> average distance across all pairwise comparisons in the cost matrix",
                               normal_style))
        ]
        bullet_list = ListFlowable(bullet_points, bulletType='bullet', start='•')
        c.append(bullet_list)
        c.append(Spacer(1, 0.2 * inch))
    algorithm_columns = [
        'Body Part', 'Metric', 'Radius', 'Normalized', 'Smooth', 'Smooth Window',
        'Distance', 'Path Length'
    ]
    matrix_columns = [
        'Body Part', 'Cost Matrix Shape', 'Min Cost', 'Max Cost', 'Mean Cost'
    ]
    algorithm_df = analysis_summary_df[algorithm_columns].copy()

    matrix_df = analysis_summary_df[matrix_columns].copy()
    c.append(create_table_from_dataframe(algorithm_df))
    c.append(Spacer(1, 0.2 * inch))
    c.append(create_table_from_dataframe(matrix_df))
    if attach_plot_images:
        c.extend(attach_images_from_folder(dtw_plots_folder, 3, padding_factor=0.75))
    return c


def generate_smith_waterman_page(normal_style, path, analysis_summary_df,
                                 explanation_text=True, attach_plot_images=True):
    similarity_plots_folder = os.path.join(path, SMITH_WATERMAN_PLOT_FOLDER_NAME)
    c = [PageBreak()]
    c.append(Paragraph("<u>Smith-Waterman Analysis:</u>", section_title_style))
    if explanation_text:
        c.append(Paragraph("Dynamic programming algorithm used for local sequence alignment - developed and used for "
                           "DNA and protein sequence comparison. Finds alignments between subsequences by identifying "
                           "regions of high similarity between two sequences and has been adapted by us to compare "
                           "time-series vectors of movements. The algorithm allows for imperfections in the alignment "
                           "with a goal of identifying best alignment between segments, highlighting regions where "
                           "two sequences exhibit similar movements, even if they occur at different times or "
                           "speeds", normal_style))
        c.append(Spacer(1, 0.2 * inch))
    c.append(Paragraph("Following table presents data gathered during Smith-Waterman analysis:", normal_style))
    c.append(Spacer(1, 0.2 * inch))
    if explanation_text:
        bullet_points = [
            ListItem(Paragraph("<b>Metric:</b> distance metric used to compare vectors", normal_style)),
            ListItem(Paragraph("<b>Match Score:</b> score assigned for similar vectors", normal_style)),
            ListItem(Paragraph("<b>Mismatch Penalty:</b> penalty assigned when vectors are not similar", normal_style)),
            ListItem(Paragraph("<b>Gap Penalty:</b> penalty for introducing a gap in the alignment", normal_style)),
            ListItem(Paragraph("<b>Threshold:</b> maximum allowable distance between vectors for them to be "
                               "considered similar", normal_style)),
            ListItem(Paragraph("<b>Alignment Score:</b> score reflecting the strength of the alignment", normal_style)),
            ListItem(Paragraph("<b>Aligned Segments:</b> number of matched vectors excluding gaps", normal_style))
        ]
        bullet_list = ListFlowable(bullet_points, bulletType='bullet', start='•')
        c.append(bullet_list)
        c.append(Spacer(1, 0.2 * inch))
    c.append(create_table_from_dataframe(analysis_summary_df))
    if attach_plot_images:
        c.extend(attach_images_from_folder(similarity_plots_folder, 2, padding_factor=0.95))
    return c


def generate_tlcc_page(normal_style, path, unified_analysis_summary_df, separate_analysis_summary_df,
                       explanation_text=True, attach_plot_images=True):
    tlcc_plots_folder = os.path.join(path, TLCC_PLOT_FOLDER_NAME)
    c = [PageBreak()]

    # Title and Introduction
    c.append(Paragraph("<u>Time-Lagged Cross-Correlation (TLCC) Analysis:</u>", section_title_style))

    if explanation_text:
        c.append(Paragraph(
            "Time-Lagged Cross-Correlation (TLCC) measures the correlation between two sequences of movement vectors "
            "over different time lags. It helps identify time-shifted similarities or synchrony between sequences, "
            "either as unified vectors or in separate dimensions (X, Y, Z). The results give insight into the "
            "relationship between the movements and allow for analysis of how they are aligned over time.",
            normal_style))
        c.append(Spacer(1, 0.2 * inch))

    # Add description for the table
    c.append(Paragraph("The following tables present data gathered during TLCC analysis:", normal_style))
    c.append(Spacer(1, 0.2 * inch))

    if explanation_text:
        bullet_points = [
            ListItem(
                Paragraph("<b>Max Lag:</b> maximum time shift considered for the TLCC analysis", normal_style)),
            ListItem(Paragraph(
                "<b>Separate Dimensions:</b> was TLCC computed for X, Y, Z separately or unified",
                normal_style)),
            ListItem(Paragraph(
                "<b>Peak Correlation:</b> highest correlation value observed",
                normal_style)),
            ListItem(Paragraph("<b>Peak Lag:</b> lag (time shift) corresponding to the peak correlation value",
                               normal_style)),
            ListItem(
                Paragraph("<b>Average Correlation:</b> average correlation value across all lags", normal_style)),
            ListItem(Paragraph(
                "<b>Correlation Std Dev:</b> standard deviation of correlation values",
                normal_style)),
            ListItem(Paragraph(
                "<b>Peak Correlation X, Y, Z:</b> highest correlation value observed for each dimension in the separate analysis",
                normal_style)),
            ListItem(Paragraph(
                "<b>Peak Lag X, Y, Z:</b> lag corresponding to the peak correlation for each dimension in the separate analysis",
                normal_style)),
            ListItem(Paragraph(
                "<b>Avg Corr X, Y, Z:</b> average correlation value for each dimension in the separate analysis",
                normal_style)),
            ListItem(Paragraph(
                "<b>Overall Avg Correlation:</b> overall average correlation across all dimensions (X, Y, Z) in the separate analysis",
                normal_style))
        ]
        bullet_list = ListFlowable(bullet_points, bulletType='bullet', start='•')
        c.append(bullet_list)
        c.append(Spacer(1, 0.2 * inch))

    c.append(Paragraph("<b>Unified TLCC Data:</b>", normal_style))
    c.append(Spacer(1, 0.2 * inch))
    c.append(create_table_from_dataframe(unified_analysis_summary_df))
    if explanation_text:
        c.append(PageBreak())
    c.append(Spacer(1, 0.2 * inch))
    c.append(Paragraph("<b>Separate TLCC Data (X, Y, Z dimensions):</b>", normal_style))
    c.append(Spacer(1, 0.2 * inch))
    c.append(create_table_from_dataframe(separate_analysis_summary_df))
    c.append(Spacer(1, 0.2 * inch))
    c.append(PageBreak())
    if attach_plot_images:
        c.extend(attach_images_from_folder(tlcc_plots_folder, 3, padding_factor=0.75))

    return c


if __name__ == "__main__":
    path = './images/test/gui_test/2024-09-16-13-03-32'
    excel_file_path = os.path.join(path, 'similarity_report_summary.xlsx')
    print(os.path.exists(excel_file_path))
    print(os.getcwd())
    similarity_report_data = pd.read_excel(excel_file_path)
    excel_file_path = os.path.join(path, 'dtw_report_summary.xlsx')
    dtw_report_data = pd.read_excel(excel_file_path)
    excel_file_path = os.path.join(path, 'smith_waterman_report_summary.xlsx')
    smith_waterman_data = pd.read_excel(excel_file_path)
    excel_file_path = os.path.join(path, 'tlcc_unified_report_summary.xlsx')
    tlcc_unified_data = pd.read_excel(excel_file_path)
    excel_file_path = os.path.join(path, 'tlcc_separate_report_summary.xlsx')
    tlcc_separate_data = pd.read_excel(excel_file_path)
    generate_intersync_report_pdf(path=path, similarity_report_df=similarity_report_data, dtw_report_df=dtw_report_data,
                                  smith_waterman_report_df=smith_waterman_data,
                                  tlcc_unified_report_df=tlcc_unified_data,
                                  tlcc_separate_report_df=tlcc_separate_data, keyframe_page_present=True,
                                  explanation_text=True, attach_plot_images=True)
