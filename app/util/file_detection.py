import os

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.read_api import read_local_images, read_local_office
local_image_dir, local_md_dir = "output/images", "output/md"
image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
    local_md_dir
)
image_dir = str(os.path.basename(local_image_dir))
def detect_pdf_content(pdf_file_name):
    """
    检测PDF文件内容
    """
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content
    dataset = PymuDocDataset(pdf_bytes)
    ## inference
    if dataset.classify() == SupportedPdfParseMethod.OCR:
        infer_result = dataset.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        infer_result = dataset.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)
    md_content = pipe_result.get_markdown(image_dir)
    return md_content
def detect_image_content(image_file_name):
    """
    检测图片文件内容
    """
    ds = read_local_images(image_file_name)[0]

    infer_result=ds.apply(doc_analyze, ocr=True)
    pipe_result = infer_result.pipe_ocr_mode(image_writer)
    md_content = pipe_result.get_markdown(image_dir)
    return md_content
def detect_office_content(office_file_name):
    """
    检测Office文件内容
    """
    ds = read_local_office(office_file_name)[0]

    infer_result=ds.apply(doc_analyze, ocr=True)
    pipe_result = infer_result.pipe_ocr_mode(image_writer)
    md_content = pipe_result.get_markdown(image_dir)
    return md_content

if __name__ == "__main__":
    print(detect_office_content("/home/dxye/Program/PhysicalExaminationAgent/client/app/uploads/1/新建 Microsoft Word 文档.docx"))