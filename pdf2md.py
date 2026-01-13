import os
import json
import copy
import glob
from loguru import logger

from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.pipe.OCRPipe import OCRPipe
from magic_pdf.pipe.TXTPipe import TXTPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
import magic_pdf.model as model_config

model_config.__use_inside_model__ = True

def json_md_dump(
        pipe,
        md_writer,
        json_writer,
        pdf_name,
        content_list,
        md_content,
):
    json_writer.write(
        content=json.dumps(pipe.model_list, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_model.json"
    )
    json_writer.write(
        content=json.dumps(pipe.pdf_mid_data, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_middle.json"
    )
    json_writer.write(
        content=json.dumps(content_list, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_content_list.json"
    )

    md_writer.write(
        content=md_content,
        path=f"{pdf_name}.md"
    )

def pdf_parse_main(
        pdf_path: str,
        parse_method: str = 'auto',
        model_json_path: str = None,
        is_json_md_dump: bool = True,
        output_dir: str = None
):
    """
    执行从 pdf 转换到 json、md 的过程，输出 md 和 json 文件到指定的目录

    :param pdf_path: .pdf 文件的路径，可以是相对路径，也可以是绝对路径
    :param parse_method: 解析方法， 共 auto、ocr、txt 三种，默认 auto，如果效果不好，可以尝试 ocr
    :param model_json_path: 已经存在的模型数据文件，如果为空则使用内置模型，pdf 和 model_json 务必对应
    :param is_json_md_dump: 是否将解析后的数据写入到 .json 和 .md 文件中，默认 True，会将不同阶段的数据写入到不同的 .json 文件中（共3个.json文件），md内容会保存到 .md 文件中
    :param output_dir: 输出结果的目录地址，不需要以 pdf 文件名命名的文件夹，只需指定主目录
    """
    try:
        pdf_name = os.path.basename(pdf_path).split(".")[0]

        if output_dir:
            json_output_path = os.path.join(output_dir, 'json')
            md_output_path = os.path.join(output_dir, 'md')
            image_output_path = os.path.join(output_dir, 'images', pdf_name)
        else:
            pdf_path_parent = os.path.dirname(pdf_path)
            json_output_path = os.path.join(pdf_path_parent, 'json')
            md_output_path = os.path.join(pdf_path_parent, 'md')
            image_output_path = os.path.join(pdf_path_parent, 'images', pdf_name)

        os.makedirs(json_output_path, exist_ok=True)
        os.makedirs(md_output_path, exist_ok=True)
        os.makedirs(image_output_path, exist_ok=True)

        image_path_parent = os.path.basename(image_output_path)

        pdf_bytes = open(pdf_path, "rb").read()

        if model_json_path:
            model_json = json.loads(open(model_json_path, "r", encoding="utf-8").read())
        else:
            model_json = []

        # 执行解析步骤
        image_writer = DiskReaderWriter(image_output_path)
        md_writer = DiskReaderWriter(md_output_path)
        json_writer = DiskReaderWriter(json_output_path)

        if parse_method == "auto":
            jso_useful_key = {"_pdf_type": "", "model_list": model_json}
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
        elif parse_method == "txt":
            pipe = TXTPipe(pdf_bytes, model_json, image_writer)
        elif parse_method == "ocr":
            pipe = OCRPipe(pdf_bytes, model_json, image_writer)
        else:
            logger.error("unknown parse method, only auto, ocr, txt allowed")
            exit(1)

        pipe.pipe_classify()

        # 如果没有传入模型数据，则使用内置模型解析
        if not model_json:
            if model_config.__use_inside_model__:
                pipe.pipe_analyze()  # 解析
            else:
                logger.error("need model list input")
                exit(1)

        pipe.pipe_parse()

        content_list = pipe.pipe_mk_uni_format(image_path_parent, drop_mode="none")
        md_content = pipe.pipe_mk_markdown(image_path_parent, drop_mode="none")

        if is_json_md_dump:
            json_md_dump(pipe, md_writer, json_writer, pdf_name, content_list, md_content)

    except Exception as e:
        logger.exception(e)

def get_md_filename(pdf_filename):
    base_name = os.path.basename(pdf_filename)  # 获取文件名（去掉路径）
    md_name = os.path.splitext(base_name)[0] + '.md'  # 替换扩展名为 .md
    return os.path.join('./ccar/md/', md_name)  # 构造出 md 文件的完整路径

if __name__ == '__main__':
    pdfs = glob.glob('./pdf/*.pdf')
    for pdf in pdfs:
        md_path = get_md_filename(pdf)
        if os.path.exists(md_path):
            print(f"文件 {md_path} 已存在，跳过转换")
            continue
        else:
            print(f"文件 {md_path} 不存在，正在转换 {pdf} 为 md...")
            pdf_parse_main(pdf, output_dir="./ccar/")
