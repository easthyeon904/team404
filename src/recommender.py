import os
import json
import pandas as pd
import networkx as nx
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class CTWPFRecommender:
    def __init__(self, api_key=None, course_file="courses.json"):
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY")
            
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Gemini Client 초기화 오류: {e}")
        
        self.keyword_cache = {}
        # 불용어 리스트
        self.STOPWORDS = {
            "강의", "수업", "학점", "교수", "시험", "과제", "평가", "출석", 
            "중간고사", "기말고사", "이해", "개요", "목표", "방법", "분석", 
            "활용", "이론", "실습", "소개", "기초", "응용", "진행", "관련",
            "학년", "전공", "필수", "선택", "학생", "사용자", "및", "의", "등", "대하여"
        }
        
        self.graph = self._load_ontology_data()
        self.target_courses = self._load_course_data(course_file)

    def _load_ontology_data(self):
        G = nx.DiGraph()
        try:
            nodes_file = "ontology.xlsx - Nodes.csv"
            edges_file = "ontology.xlsx - Edges.csv"
            
            if not os.path.exists(nodes_file) or not os.path.exists(edges_file):
                return None

            nodes_df = pd.read_csv(nodes_file)
            edges_df = pd.read_csv(edges_file)
            
            for _, row in nodes_df.iterrows():
                node_id = str(row['id']).strip()
                label = str(row['label']).strip()
                G.add_node(node_id, label=label, type=str(row.get('mode', '')), description=str(row.get('description', '')))
                
            for _, row in edges_df.iterrows():
                src = str(row['sourceID']).strip()
                dst = str(row['targetID']).strip()
                rel = str(row['relation']).strip()
                G.add_edge(src, dst, relation=rel)
            return G
        except Exception:
            return None

    def _load_course_data(self, file_path):
        try:
            if not os.path.exists(file_path):
                return []
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception:
            return []

    def get_course_data(self, course_name):
        for course in self.target_courses:
            if course.get('course_name') == course_name:
                return course
        return {"keywords": {}, "description": ""}

    def get_keywords_from_input(self, major, double_major, history_courses, interest):
        keywords = {}
        input_items = [major]
        if double_major and double_major != "없음":
            input_items.append(double_major)
        if history_courses:
            input_items.extend(history_courses)
        
        if interest:
            keywords[interest] = 1.0

        if self.graph is None:
            for item in input_items: keywords[item] = 1.0
            return keywords

        # 온톨로지 탐색
        for item in input_items:
            keywords[item] = 1.0
            target_nodes = []
            
            for node_id, data in self.graph.nodes(data=True):
                label = data.get('label', '')
                if item.replace(" ", "") in label.replace(" ", ""):
                    target_nodes.append(node_id)
            
            for root_node in target_nodes:
                neighbors = list(self.graph.successors(root_node)) + list(self.graph.predecessors(root_node))
                for neighbor in neighbors:
                    neighbor_label = self.graph.nodes[neighbor].get('label', '')
                    if neighbor_label and neighbor_label not in self.STOPWORDS:
                        keywords[neighbor_label] = keywords.get(neighbor_label, 0) + 0.5
        return keywords

    def get_idf_weight(self, word):
        return 1.5 if len(word) > 2 else 1.0

    def expand_keyword_with_gemini(self, keyword):
        """WP 알고리즘용 키워드 확장"""
        if not self.client: return []
        if keyword in self.keyword_cache: return self.keyword_cache[keyword]
        
        try:
            if keyword in self.STOPWORDS: return []
            
            prompt = f"""
            키워드: "{keyword}"
            이 키워드의 상위 개념과 연관된 학술/직무 용어 5개를 쉼표로 구분하여 나열하라. 설명 금지.
            """
            
            response = self.client.models.generate_content(
                model='gemini-2.0-flash', 
                contents=prompt
            )
            
            if not response.text: return []
            related_words = [word.strip() for word in response.text.split(',')]
            self.keyword_cache[keyword] = related_words
            return related_words
        except Exception:
            return []

    def evaluate_relevance_with_gemini(self, student_info, course_info):
        """
        [NEW] Gemini가 직접 학생과 강의의 적합도를 평가하고 사유를 생성
        """
        if not self.client: return 0, "API 오류"
        
        prompt = f"""
        당신은 대학 수강신청 컨설턴트입니다. 아래 학생 정보와 강의 정보를 비교하여 추천 적합도를 평가해주세요.

        [학생 정보]
        - 전공/이수과목 키워드: {student_info.get('profile_keywords')}
        - 관심분야: {student_info.get('interest')}

        [강의 정보]
        - 강의명: {course_info.get('course_name')}
        - 강의개요: {course_info.get('description')}
        - 핵심키워드: {list(course_info.get('keywords', {}).keys())}

        [지시사항]
        1. 학생의 관심사 및 전공 배경지식이 이 강의를 수강하는 데 얼마나 도움이 되거나, 목표에 부합하는지 분석하십시오.
        2. 적합도 점수를 0점에서 50점 사이의 정수로 매기십시오. (50점 만점)
        3. 추천하는 핵심 이유를 1문장으로 요약하여 작성하십시오.
        4. 응답은 반드시 JSON 형식으로 작성하십시오: {{"score": 점수, "reason": "이유"}}
        """

        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            result = json.loads(response.text)
            return result.get("score", 0), result.get("reason", "분석 불가")
        except Exception as e:
            # print(f"Gemini 평가 오류: {e}") 
            return 0, "AI 분석 실패"

    def calculate_ctwp_score(self, student_dict, course_keywords):
        """CTWP 알고리즘 (기초 적합도)"""
        common_terms = set(student_dict.keys()).intersection(set(course_keywords.keys()))
        filtered_ct = common_terms - self.STOPWORDS
        
        ct_score = 0
        matches = []
        for term in filtered_ct:
            s_weight = student_dict.get(term, 1.0)
            c_weight = course_keywords.get(term, 1.0)
            ct_score += (s_weight * c_weight * self.get_idf_weight(term))
            matches.append(term)

        wp_score = 0
        # 상위 5개만 확장
        sorted_student_terms = sorted(student_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for s_term, s_weight in sorted_student_terms:
            if s_term in self.STOPWORDS: continue
            expanded = self.expand_keyword_with_gemini(s_term)
            for e_term in expanded:
                if e_term in course_keywords:
                    if e_term in self.STOPWORDS or e_term in matches: continue
                    c_weight = course_keywords.get(e_term, 1.0)
                    wp_score += (s_weight * 0.8 * c_weight)
                    matches.append(f"{e_term}(←{s_term})")
        
        final_score = (0.5 * ct_score) + (0.5 * wp_score)
        return final_score * 5.0, list(set(matches))

    def run_analysis(self, major, double_major, history, interest):
        """[Main] Top 5 추천 실행"""
        
        student_profile = self.get_keywords_from_input(major, double_major, history, interest)
        
        # Gemini 평가용 학생 정보 텍스트 구성
        student_info_text = {
            "profile_keywords": ", ".join(list(student_profile.keys())[:10]),
            "interest": interest if interest else "없음"
        }
        
        recommendations = []
        
        for course in self.target_courses:
            course_keywords = course.get('keywords', {})
            
            # (1) CTWP 점수 (키워드 매칭)
            ctwp_score, matches = self.calculate_ctwp_score(student_profile, course_keywords)
            
            # (2) Gemini 직접 평가 (문맥/의미 매칭)
            # API 호출 비용/시간 고려: CTWP 점수가 너무 낮지 않거나, 관심분야가 명확할 때 수행하면 좋지만
            # 여기서는 정확도를 위해 모든 대상(또는 상위 N개)에 대해 수행
            
            # (단, 14개 강의 전체 수행 시 시간 소요됨. 여기서는 전체 수행)
            ai_score, ai_reason = self.evaluate_relevance_with_gemini(student_info_text, course)
            
            # (3) 최종 점수 합산
            total_score = ctwp_score + ai_score
            
            recommendations.append({
                "강의명": course.get('course_name', 'Unknown'),
                "교수": course.get('professor', 'Unknown'),
                "university": course.get('university', '타대학'),
                "최종 점수": total_score,
                "CTWP 점수": ctwp_score,
                "AI 점수": ai_score,
                "추천 사유": ai_reason, # Gemini가 작성한 이유
                "매칭 키워드": ", ".join(matches[:5]) if matches else "없음"
            })
            
        df = pd.DataFrame(recommendations)
        if not df.empty:
            # 점수 백분율 환산
            max_score = df["최종 점수"].max()
            if max_score > 0:
                df["적합도(%)"] = (df["최종 점수"] / max_score) * 95
                df["적합도(%)"] = df["적합도(%)"].astype(int)
            else:
                df["적합도(%)"] = 0
                
            df = df.sort_values(by="최종 점수", ascending=False).head(5)
        
        return df, student_profile