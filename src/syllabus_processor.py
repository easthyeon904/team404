import os
import json
import glob
import time
import logging
import re
import pdfplumber
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# pdfminer 경고 로그 숨기기
logging.getLogger("pdfminer").setLevel(logging.ERROR)

class SyllabusProcessor:
    def __init__(self, input_folder="syllabus_files", output_file="courses.json"):
        self.input_folder = input_folder
        self.output_file = output_file
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            print("⚠️ 경고: GEMINI_API_KEY가 설정되지 않았습니다.")
            
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Gemini Client 초기화 오류: {e}")

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text
        except Exception as e:
            return ""

    def get_embedding(self, text):
        if not self.client: return []
        try:
            result = self.client.models.embed_content(
                model="text-embedding-004",
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            return result.embeddings[0].values
        except Exception as e:
            return []

    def extract_university_from_filename(self, filename):
        """
        [NEW] 파일명에서 학교명 추출 로직
        예: '화공유체_서울대.pdf' -> '서울대'
        예: '프로젝트_기획_한예종.pdf' -> '한예종'
        예: '디지털지도학(시립대).pdf' -> '시립대'
        """
        # 확장자 제거
        name = os.path.splitext(filename)[0]
        
        # 1. 폴더명으로 구분된 경우 (예: 서울대예시/화공유체.pdf)
        # -> process_all_files에서 path를 넘겨받아 처리하는게 좋지만,
        # 여기서는 파일명 자체에 힌트가 있다고 가정하고 패턴 매칭
        
        university = "타대학" # 기본값
        
        # 패턴 1: 언더바(_) 뒤에 학교명이 있는 경우 (가장 마지막 _ 뒤)
        if "_" in name:
            parts = name.split("_")
            potential_uni = parts[-1]
            # 학교명 리스트 확인 (서울대, 시립대, 한예종 등)
            if any(univ in potential_uni for univ in ["서울대", "시립대", "한예종", "연세대", "고려대"]):
                university = potential_uni
        
        # 패턴 2: 괄호 안에 학교명이 있는 경우
        match = re.search(r'\((.*?)\)', name)
        if match:
            content = match.group(1)
            if any(univ in content for univ in ["서울대", "시립대", "한예종"]):
                university = content
                
        # [특수 처리] 압축파일 폴더 구조 힌트 사용
        # 사용자가 올린 파일 경로에 힌트가 있다면 그것을 우선할 수도 있음.
        # 여기서는 파일명 패턴이 가장 확실하다고 보고 진행.
        
        # 파일명에 명시적으로 학교 이름이 포함된 경우
        if "서울대" in name: return "서울대"
        if "시립대" in name: return "서울시립대"
        if "한예종" in name: return "한예종"
        
        return university

    def analyze_content_with_gemini(self, raw_text, file_name):
        if not self.client:
            return {"course_name": file_name, "keywords": {}, "description": "API Key 없음"}

        prompt = f"""
        당신은 '교육 공학 전문가'입니다. 아래 대학 강의계획서 텍스트를 분석하여 JSON 데이터를 추출하십시오.
        
        [필수 포함 필드]
        1. course_name: 강의명 (문자열, 정확하게)
        2. professor: 교수명 (문자열, 없으면 "미상")
        3. description: 강의 개요 및 목표 (문자열, 3문장 요약)
        4. keywords: 핵심 역량 키워드 5~10개와 가중치 (예: {{"마케팅": 1.0, "분석": 0.8}})
        5. 4c_id_components: {{"learning_tasks": ["과제1", "과제2"]}}

        * 주의: 응답은 반드시 JSON 객체({{...}})여야 합니다.

        [강의계획서 텍스트]
        {raw_text[:20000]}
        """

        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            data = json.loads(response.text)
            
            if isinstance(data, list):
                data = data[0] if len(data) > 0 else {}
            
            if not isinstance(data, dict):
                return {"course_name": file_name, "keywords": {}, "description": "형식 오류"}
                
            return data
            
        except Exception as e:
            print(f"  ⚠️ LLM 분석 오류 ({file_name}): {e}")
            return {"course_name": file_name, "description": "분석 실패", "keywords": {}}

    def process_all_files(self):
        # 하위 폴더까지 모두 검색
        pdf_files = glob.glob(os.path.join(self.input_folder, "**/*.pdf"), recursive=True)
        
        if not pdf_files:
            print(f"⚠️ '{self.input_folder}' 폴더 내에서 PDF 파일을 찾을 수 없습니다.")
            return

        print(f"📄 총 {len(pdf_files)}개의 PDF 파일을 발견했습니다. 분석을 시작합니다...")

        processed_data = []

        for i, pdf_path in enumerate(pdf_files):
            file_name = os.path.basename(pdf_path)
            # 폴더 경로에서도 힌트 얻기 (예: syllabus_files/서울대예시/파일.pdf)
            full_path_str = str(pdf_path)
            
            print(f"[{i+1}/{len(pdf_files)}] 분석 중: {file_name} ...")
            
            # 1. 학교명 추출 (파일명 우선, 없으면 경로명)
            university = "타대학"
            if "서울대" in full_path_str: university = "서울대"
            elif "시립대" in full_path_str: university = "서울시립대"
            elif "한예종" in full_path_str: university = "한예종"
            else:
                university = self.extract_university_from_filename(file_name)
            
            # 2. 텍스트 추출
            raw_text = self.extract_text_from_pdf(pdf_path)
            
            if not raw_text.strip():
                print(f"  ⚠️ 텍스트 추출 실패: {file_name}")
                continue

            # 3. LLM 분석
            structured_data = self.analyze_content_with_gemini(raw_text, file_name)
            
            if isinstance(structured_data, dict):
                structured_data["id"] = f"C{str(i+1).zfill(3)}"
                structured_data["filename"] = file_name
                structured_data["university"] = university # [NEW] 학교명 필드 추가
                
                # 임베딩 생성용 텍스트
                context_text = f"{structured_data.get('course_name', '')} {structured_data.get('description', '')} "
                keywords = structured_data.get('keywords', {})
                if isinstance(keywords, dict):
                    context_text += " ".join(keywords.keys())
                
                structured_data["embedding"] = self.get_embedding(context_text)
                processed_data.append(structured_data)
            else:
                print(f"  ❌ 데이터 구조 오류: {file_name}")

            time.sleep(1)

        # JSON 저장
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
        print(f"\n✅ 분석 완료! 총 {len(processed_data)}개의 강의 데이터가 '{self.output_file}'에 저장되었습니다.")

if __name__ == "__main__":
    if os.getenv("GEMINI_API_KEY"):
        processor = SyllabusProcessor()
        processor.process_all_files()
    else:
        print("❌ .env 파일 확인 필요")