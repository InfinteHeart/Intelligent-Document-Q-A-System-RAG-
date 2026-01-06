import requests
import time
import zipfile
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
api_key = ALIYUN_API_KEY

# ==== 阿里云OSS配置 ====
ALIYUN_OSS_BUCKET = Bucket_name  # 修改为您的Bucket名称
ALIYUN_OSS_ENDPOINT = OSS_ENDPOINT  # 修改为自己的区域
ALIYUN_OSS_BASE_URL = f'https://{ALIYUN_OSS_BUCKET}.{ALIYUN_OSS_ENDPOINT.replace("https://", "")}/'  # 自动生成Base URL

def upload_file_to_temp_storage(file_path):
    """
    将文件上传到临时存储并返回可访问的URL
    使用阿里云OSS作为临时存储
    """
    file_path = Path(file_path)
    file_name = file_path.name
    
    try:
        import oss2
        
        logger.info(f"尝试上传文件到OSS: {file_name}")
        
        # ==== 设置OSS访问凭证 ====
        # 方法1：从环境变量获取（推荐）
        access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
        access_key_secret = os.getenv('OSS_ACCESS_KEY_SECRET')
        
        # 方法2：如果环境变量不存在，使用硬编码（仅用于测试）
        if not access_key_id or not access_key_secret:
            logger.warning("使用环境变量获取OSS凭证失败，请设置环境变量 OSS_ACCESS_KEY_ID 和 OSS_ACCESS_KEY_SECRET")
            # 在这里填入您的AccessKey（仅用于测试，生产环境请使用环境变量）
            access_key_id = '您的AccessKeyId'
            access_key_secret = '您的AccessKeySecret'
        
        # 验证凭证是否设置
        if not access_key_id or not access_key_secret or access_key_id == '您的AccessKeyId':
            raise ValueError("未设置有效的阿里云OSS访问凭证。请设置环境变量 OSS_ACCESS_KEY_ID 和 OSS_ACCESS_KEY_SECRET")
        
        # 创建认证对象
        auth = oss2.Auth(access_key_id, access_key_secret)
        
        # 创建Bucket对象
        bucket = oss2.Bucket(auth, ALIYUN_OSS_ENDPOINT, ALIYUN_OSS_BUCKET)
        
        # ==== 上传文件到根目录（不放到pdf子目录） ====
        object_name = file_name  # 直接使用文件名，不添加'pdf/'前缀
        
        with open(file_path, 'rb') as f:
            result = bucket.put_object(object_name, f)
        
        if result.status == 200:
            # 设置文件为公共读权限
            bucket.put_object_acl(object_name, oss2.OBJECT_ACL_PUBLIC_READ)
            logger.info(f"已成功设置文件公共读权限")
            
            file_url = f"{ALIYUN_OSS_BASE_URL}{file_name}"
            logger.info(f"文件上传成功: {file_url}")
            return file_url
        else:
            logger.error(f"文件上传失败，状态码: {result.status}")
            raise RuntimeError(f"文件上传失败，状态码: {result.status}")
        
    except ImportError:
        logger.error("未安装阿里云OSS SDK，请执行: pip install oss2")
        raise RuntimeError("请先安装阿里云OSS SDK: pip install oss2")
    except Exception as e:
        logger.error(f"上传到OSS失败: {e}")
        raise RuntimeError(f"上传到临时存储失败: {e}")

def get_task_id(file_path):
    """
    使用URL方式提交解析任务（兼容性更好）
    步骤：
    1. 将文件上传到临时存储并获取可访问的URL
    2. 使用URL提交解析任务
    """
    file_path = Path(file_path)
    
    try:
        # 步骤1: 将文件上传到临时存储并获取可访问的URL
        upload_url = upload_file_to_temp_storage(file_path)
        
        # 步骤2: 使用URL提交解析任务
        task_url = "https://mineru.net/api/v4/extract/task"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 使用URL方式提交，包含必要的参数
        data = {
            "url": upload_url,
            "is_ocr": True,
            "enable_formula": False
        }
        
        logger.info(f"提交解析任务，URL: {upload_url}")
        response = requests.post(task_url, headers=headers, json=data, timeout=30)
        
        try:
            result = response.json()
            logger.debug(f"解析任务响应: {result}")
        except requests.exceptions.JSONDecodeError:
            logger.error(f"解析任务响应不是有效的JSON格式: {response.text}")
            raise RuntimeError(f"解析任务响应不是有效的JSON格式: {response.text}")
        
        if result["code"] != 0:
            logger.error(f"提交解析任务失败: {result.get('msg', '未知错误')}")
            raise RuntimeError(f"提交解析任务失败: {result.get('msg', '未知错误')}")
        
        task_id = result["data"]["task_id"]
        logger.info(f"成功获取task_id: {task_id}")
        return task_id
        
    except Exception as e:
        logger.error(f"获取task_id时发生错误: {e}")
        raise RuntimeError(f"获取task_id失败: {e}")

def get_result(task_id):
    """
    获取解析结果
    """
    url = f'https://mineru.net/api/v4/extract/task/{task_id}'
    header = {
        'Content-Type':'application/json',
        "Authorization":f"Bearer {api_key}"
    }

    max_retries = 60  # 最大重试次数（60*10=600秒=10分钟）
    retry_count = 0
    wait_time = 10  # 等待时间（秒）

    logger.info(f"开始获取任务结果，任务ID: {task_id}")
    logger.info(f"将尝试 {max_retries} 次，每次等待 {wait_time} 秒")

    while retry_count < max_retries:
        try:
            logger.debug(f"第 {retry_count+1}/{max_retries} 次尝试获取结果...")
            response = requests.get(url, headers=header, timeout=30)
            
            try:
                res_json = response.json()
                logger.debug(f"获取结果响应: {res_json}")
            except requests.exceptions.JSONDecodeError:
                logger.error(f"获取结果响应不是有效的JSON格式: {response.text}")
                raise RuntimeError(f"获取结果响应不是有效的JSON格式: {response.text}")
            
            # 检查API响应是否成功
            if response.status_code != 200 or res_json.get('code') != 0:
                logger.error(f"获取结果API请求失败: {res_json.get('msg', '未知错误')}")
                raise RuntimeError(f"获取结果API请求失败: {res_json.get('msg', '未知错误')}")
            
            # 检查是否存在data字段
            if 'data' not in res_json:
                logger.error("获取结果API响应中缺少'data'字段")
                raise RuntimeError("获取结果API响应中缺少'data'字段")
            
            result = res_json["data"]
            state = result.get('state')
            err_msg = result.get('err_msg', '')
            
            # 如果有错误，输出错误信息
            if err_msg:
                logger.error(f"任务出错: {err_msg}")
                raise RuntimeError(f"任务出错: {err_msg}")
            
            # 如果任务还在进行中，等待后重试
            if state in ['pending', 'running']:
                elapsed_time = retry_count * wait_time
                task_progress = result.get('progress', '未知')
                logger.info(f"任务未完成（状态: {state}，进度: {task_progress}%，已等待: {elapsed_time}秒），等待{wait_time}秒后重试...")
                logger.debug(f"完整任务信息: {result}")
                time.sleep(wait_time)
                retry_count += 1
                continue
            
            # 如果任务完成，下载文件
            if state == 'done':
                full_zip_url = result.get('full_zip_url')
                if full_zip_url:
                    local_filename = f"{task_id}.zip"
                    logger.info(f"任务完成，开始下载结果: {full_zip_url}")
                    
                    # 下载文件
                    r = requests.get(full_zip_url, stream=True, timeout=60)
                    with open(local_filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    logger.info(f"下载完成，已保存到: {local_filename}")
                    # 下载完成后自动解压
                    unzip_file(local_filename)
                    return True
                else:
                    logger.error("未找到 full_zip_url，无法下载。")
                    raise RuntimeError("未找到 full_zip_url，无法下载。")
            
            # 其他未知状态
            logger.error(f"未知状态: {state}")
            raise RuntimeError(f"未知状态: {state}")
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"网络连接错误: {e}")
            time.sleep(5)
            retry_count += 1
        except requests.exceptions.Timeout as e:
            logger.error(f"API请求超时: {e}")
            time.sleep(5)
            retry_count += 1
        except Exception as e:
            logger.error(f"获取结果时发生未知错误: {e}")
            raise
    
    logger.error(f"获取结果超时，已重试 {max_retries} 次")
    raise RuntimeError(f"获取结果超时，已重试 {max_retries} 次")

# 解压zip文件的函数
def unzip_file(zip_path, extract_dir=None):
    """
    解压指定的zip文件到目标文件夹。
    :param zip_path: zip文件路径
    :param extract_dir: 解压目标文件夹，默认为zip同名目录
    """
    zip_path = Path(zip_path)
    if extract_dir is None:
        extract_dir = zip_path.parent / zip_path.stem
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info(f"已解压到: {extract_dir}")
        return extract_dir
    except Exception as e:
        logger.error(f"解压文件失败: {e}")
        raise RuntimeError(f"解压文件失败: {e}")

if __name__ == "__main__":
    """
    测试函数
    """
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'xxx.pdf'
    
    task_id = get_task_id(file_path)
    logger.info(f'成功获取task_id: {task_id}')
    get_result(task_id)