import streamlit as st
from pathlib import Path
from src.pipeline import SinglePDFPipeline
import json
import re
import os

st.set_page_config(page_title="智能文档问答系统", layout="wide")

def extract_json_from_string(text):
    if isinstance(text, str):
        json_match = re.search(r'```json\s*(\{.*?\})\s*```|```\s*(\{.*?\})\s*```|\{.*?\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1) or json_match.group(2) or json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                json_str = json_str.strip()
                if json_str.startswith('```'):
                    json_str = json_str[3:]
                if json_str.endswith('```'):
                    json_str = json_str[:-3]
                json_str = json_str.strip()
                try:
                    return json.loads(json_str)
                except:
                    return None
    return None

def format_answer(answer):
    step_by_step = "-"
    reasoning_summary = "-"
    relevant_pages = []
    final_answer = "-"
    
    if isinstance(answer, dict):
        if "final_answer" in answer and isinstance(answer["final_answer"], str):
            json_data = extract_json_from_string(answer["final_answer"])
            if json_data:
                step_by_step = json_data.get("step_by_step_analysis", 
                                           answer.get("step_by_step_analysis", "-"))
                reasoning_summary = json_data.get("reasoning_summary", 
                                                 answer.get("reasoning_summary", "-"))
                relevant_pages = json_data.get("relevant_pages", 
                                               answer.get("relevant_pages", []))
                final_answer = json_data.get("final_answer", 
                                           answer.get("final_answer", "-"))
            else:
                step_by_step = answer.get("step_by_step_analysis", "-")
                reasoning_summary = answer.get("reasoning_summary", "-")
                relevant_pages = answer.get("relevant_pages", [])
                final_answer = answer.get("final_answer", "-")
        else:
            step_by_step = answer.get("step_by_step_analysis", "-")
            reasoning_summary = answer.get("reasoning_summary", "-")
            relevant_pages = answer.get("relevant_pages", [])
            final_answer = answer.get("final_answer", "-")
            
    elif isinstance(answer, str):
        try:
            answer_dict = json.loads(answer)
            if isinstance(answer_dict, dict):
                json_data = extract_json_from_string(answer)
                if json_data:
                    step_by_step = json_data.get("step_by_step_analysis", "-")
                    reasoning_summary = json_data.get("reasoning_summary", "-")
                    relevant_pages = json_data.get("relevant_pages", [])
                    final_answer = json_data.get("final_answer", "-")
                else:
                    final_answer = answer
        except json.JSONDecodeError:
            json_data = extract_json_from_string(answer)
            if json_data:
                step_by_step = json_data.get("step_by_step_analysis", "-")
                reasoning_summary = json_data.get("reasoning_summary", "-")
                relevant_pages = json_data.get("relevant_pages", [])
                final_answer = json_data.get("final_answer", "-")
            else:
                final_answer = answer
    
    if step_by_step in ["-", "", None, "null"] or (isinstance(step_by_step, str) and not step_by_step.strip()):
        step_by_step = "无分步推理内容"
    if reasoning_summary in ["-", "", None, "null"] or (isinstance(reasoning_summary, str) and not reasoning_summary.strip()):
        reasoning_summary = "无推理摘要内容"
    if final_answer in ["-", "", None, "null"] or (isinstance(final_answer, str) and not final_answer.strip()):
        final_answer = "无最终答案"
    
    if not isinstance(relevant_pages, list):
        if isinstance(relevant_pages, (int, float)):
            relevant_pages = [relevant_pages]
        else:
            relevant_pages = []
    
    return step_by_step, reasoning_summary, relevant_pages, final_answer

def display_answer_result(step_by_step, reasoning_summary, relevant_pages, final_answer):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**分步推理：**")
        st.info(step_by_step)
        
        st.markdown("**相关页面：**")
        if relevant_pages:
            for i, page in enumerate(relevant_pages):
                st.write(f"- 第{page}页")
        else:
            st.write("无相关页面信息")
    
    with col2:
        st.markdown("**推理摘要：**")
        st.success(reasoning_summary)
        
        st.markdown("**最终答案：**")
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 12px;
            border-left: 6px solid #7b2ff2;
            box-shadow: 0 4px 12px rgba(123, 47, 242, 0.1);
            font-size: 16px;
            line-height: 1.6;
        '>
            {final_answer}
        </div>
        """, unsafe_allow_html=True)

def main():
    st.markdown("""
    <div style='background: linear-gradient(90deg, #7b2ff2 0%, #f357a8 100%); padding: 20px; border-radius: 12px; text-align: center;'>
        <h2 style='color: white; margin: 0;'>🤖 智能问答系统</h2>
        <div style='color: #fff; font-size: 16px;'>上传文档，智能问答从此简单</div>
    </div>
    """, unsafe_allow_html=True)

    # 初始化处理状态
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    st.markdown("### 选择使用模式")
    
    mode = st.radio(
        "请选择模式：",
        ["通用问答", "公司年报问答", "学习资料问答", "股票投资问答", "汽车领域问答", "医疗健康问答"],
        horizontal=True,
        disabled=st.session_state.processing  # 处理中禁用切换
    )
    
    # 初始化领域特定的会话状态存储
    if 'domain_pipelines' not in st.session_state:
        st.session_state.domain_pipelines = {}  # 存储不同领域的pipeline实例
    if 'domain_uploaded_files' not in st.session_state:
        st.session_state.domain_uploaded_files = {}  # 存储不同领域的上传文件
    


    # 处理所有垂直领域的PDF问答模式，包括公司年报问答
    domain_map = {
        "通用问答": "universal",
        "公司年报问答": "annual_report",
        "学习资料问答": "education",
        "股票投资问答": "stock",
        "汽车领域问答": "automotive",
        "医疗健康问答": "medical"
    }
    domain = domain_map[mode]
    
    st.markdown("---")
    domain_info = {
        "通用问答": "🎯 通用问答模式：上传任意文档，即时问答",
        "公司年报问答": "💼 公司年报问答模式：上传公司年报、财务报表等文档",
        "学习资料问答": "📚 学习资料问答模式：上传教材、讲义等学习资料",
        "股票投资问答": "📈 股票投资问答模式：上传股票报告、财务数据等投资资料",
        "汽车领域问答": "🚗 汽车领域问答模式：上传汽车说明书、维修手册等资料",
        "医疗健康问答": "🏥 医疗健康问答模式：上传医学书籍、诊断指南等资料"
    }
    st.success(domain_info[mode])
    
    # 确保当前领域的存储存在
    if domain not in st.session_state.domain_pipelines:
        st.session_state.domain_pipelines[domain] = None
    if domain not in st.session_state.domain_uploaded_files:
        st.session_state.domain_uploaded_files[domain] = []
    
    # 获取当前领域的pipeline和上传文件
    pdf_pipeline = st.session_state.domain_pipelines[domain]
    uploaded_files = st.session_state.domain_uploaded_files[domain]
    
    with st.sidebar:
        st.header("📤 PDF文件上传")
        # 使用不同的变量名避免冲突
        new_uploaded_files = st.file_uploader("选择PDF文件（可多选）", type=['pdf'], accept_multiple_files=True)
        
        if new_uploaded_files:
            if st.button("📁 上传并处理", use_container_width=True, disabled=st.session_state.processing):
                st.session_state.processing = True  # 设置处理状态为True
                with st.spinner("正在解析PDF并建立索引..."):
                    try:
                        save_dir = Path("data/uploaded_pdfs")
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        # 确保创建的pipeline与当前领域匹配
                        if pdf_pipeline is None or pdf_pipeline.domain != domain:
                            if pdf_pipeline:
                                pdf_pipeline.clear()
                            pdf_pipeline = SinglePDFPipeline(domain=domain)
                            uploaded_files = []  # 重置已上传文件列表
                            # 更新会话状态
                            st.session_state.domain_pipelines[domain] = pdf_pipeline
                            st.session_state.domain_uploaded_files[domain] = uploaded_files
                        
                        for uploaded_file in new_uploaded_files:
                            file_path = save_dir / uploaded_file.name
                            with open(file_path, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                            
                            result = pdf_pipeline.upload_pdf(
                                str(file_path), 
                                document_name=uploaded_file.name
                            )
                            
                            if result.get("status") == "success":
                                uploaded_files.append(result)
                                st.session_state.domain_uploaded_files[domain] = uploaded_files
                                st.success(f"✅ {uploaded_file.name} 处理完成！")
                            else:
                                st.error(f"❌ {uploaded_file.name} 处理失败: {result}")
                            
                    except Exception as e:
                        st.error(f"处理PDF时出错: {e}")
                        import traceback
                        st.error(f"详细错误: {traceback.format_exc()}")
                    finally:
                        st.session_state.processing = False  # 处理完成后重置状态
        
        st.markdown("---")
        st.header("📚 已上传文档")
        
        if uploaded_files:
            for i, doc in enumerate(uploaded_files):
                st.markdown(f"""
                <div style='background: #f0f2f6; padding: 10px; border-radius: 8px; margin: 5px 0;'>
                    <strong>📄 {doc.get('document_name', doc.get('filename', 'Unknown'))}</strong><br>
                    <small>分块数: {doc.get('chunks_count', 'N/A')}</small>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🗑️ 清空所有文档", use_container_width=True):
                if pdf_pipeline:
                    pdf_pipeline.clear()
                st.session_state.domain_pipelines[domain] = None
                st.session_state.domain_uploaded_files[domain] = []
                st.rerun()
        else:
            st.write("暂无上传的文档")
    
    st.markdown("<h3 style='margin-top: 24px;'>💬 智能问答</h3>", unsafe_allow_html=True)
    
    user_question = st.text_area("输入您的问题", height=80, 
                                 placeholder="例如：这篇文档的主要内容是什么？",
                                 key=f"question_{domain}")
    
    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        answer_type = st.selectbox("答案类型", ["string", "number", "boolean", "names"])
    with col_q2:
        st.markdown("<br>", unsafe_allow_html=True)
        ask_btn = st.button("🔍 提问", use_container_width=True)
    
    if ask_btn and user_question.strip():
        if not uploaded_files:
            st.error("❌ 请先上传并处理PDF文档")
        else:
            with st.spinner("正在分析问题并检索相关内容..."):
                try:
                    answer = pdf_pipeline.answer_question(
                        user_question, 
                        kind=answer_type
                    )
                    
                    step_by_step, reasoning_summary, relevant_pages, final_answer = format_answer(answer)
                    display_answer_result(step_by_step, reasoning_summary, relevant_pages, final_answer)
                    
                except Exception as e:
                    st.error(f"生成答案时出错: {e}")
                    import traceback
                    st.error(f"详细错误信息: {traceback.format_exc()}")
    elif not uploaded_files:
        st.info("👆 请在左侧上传PDF文件，然后开始问答")
    else:
        st.info("💭 请输入问题并点击【提问】按钮")

if __name__ == "__main__":
    main()