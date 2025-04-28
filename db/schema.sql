-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 의료 문헌 데이터 테이블
CREATE TABLE medical_literature (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT[] NOT NULL,
    publication_date DATE,
    journal TEXT,
    abstract TEXT,
    full_text TEXT,
    doi TEXT UNIQUE,
    mesh_terms TEXT[],
    embedding VECTOR(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 임상 가이드라인 테이블
CREATE TABLE clinical_guidelines (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    organization TEXT NOT NULL,
    publish_date DATE,
    update_date DATE,
    specialty TEXT,
    recommendation_level TEXT,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 환자 데이터 테이블
CREATE TABLE patient_records (
    id BIGSERIAL PRIMARY KEY,
    patient_id TEXT UNIQUE NOT NULL,
    demographics JSONB NOT NULL,
    medical_history JSONB,
    medications JSONB,
    lab_results JSONB,
    vitals JSONB,
    notes TEXT,
    embedding VECTOR(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 의료 이미지 데이터 테이블
CREATE TABLE medical_images (
    id BIGSERIAL PRIMARY KEY,
    patient_id TEXT NOT NULL,
    study_id TEXT,
    modality TEXT NOT NULL,
    body_part TEXT,
    acquisition_date TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    storage_path TEXT NOT NULL,
    embedding VECTOR(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 벡터 검색을 위한 인덱스 생성
CREATE INDEX medical_literature_embedding_idx ON medical_literature USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);
CREATE INDEX clinical_guidelines_embedding_idx ON clinical_guidelines USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX patient_records_embedding_idx ON patient_records USING ivfflat (embedding vector_cosine_ops) WITH (lists = 500);
CREATE INDEX medical_images_embedding_idx ON medical_images USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200); 