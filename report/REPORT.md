# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Mai Đức Thuận
**MSSV:** 2A202600125
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

Cosine similarity đo góc giữa hai vector. Giá trị gần 1.0 nghĩa là hai vector cùng hướng, tức hai đoạn văn bản có phân bố từ vựng hoặc ngữ nghĩa tương tự nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Python là ngôn ngữ lập trình phổ biến"
- Sentence B: "Python programming language is popular"
- Lý do: Cùng nhắc đến Python là ngôn ngữ lập trình phổ biến, chỉ khác ngôn ngữ biểu đạt. Kết quả thực tế: 0.9300

**Ví dụ LOW similarity:**
- Sentence A: "Chương trình có 3 giai đoạn đào tạo"
- Sentence B: "Món phở Hà Nội rất ngon"
- Lý do: Hoàn toàn khác chủ đề, không có từ vựng chung. Kết quả thực tế: 0.6596 (vẫn khá cao do mock embedder dùng MD5 hash, không phải semantic thật sự)

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

Cosine similarity đo hướng của vector, không phụ thuộc độ dài. Hai văn bản dài ngắn khác nhau nhưng cùng nội dung vẫn có cosine cao. Euclidean distance bị ảnh hưởng bởi độ dài vector, nên văn bản dài hơn sẽ có distance lớn dù nội dung giống nhau.

### Chunking Math (Ex 1.2)

**Document 10.000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

Step = chunk_size - overlap = 500 - 50 = 450
num_chunks = ceil((10000 - 50) / 450) = ceil(9950 / 450) = ceil(22.11) = 23 chunks

**Nếu overlap tăng lên 100 thì sao? Tại sao muốn overlap nhiều hơn?**

Step = 500 - 100 = 400. num_chunks = ceil((10000 - 100) / 400) = ceil(9900 / 400) = ceil(24.75) = 25 chunks. Tăng 2 chunks.

Overlap nhiều hơn giúp thông tin ở cuối chunk trước xuất hiện cả ở đầu chunk sau, tránh mất ngữ cảnh khi cắt. Đặc biệt hữu ích khi chunk cắt ngang giữa câu.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Hướng dẫn và FAQ chương trình Đào tạo Nhân tài AI Thực chiến — VinUniversity

**Lý do chọn:** Tài liệu có cấu trúc rõ ràng, nội dung tiếng Việt đa dạng gồm thông tin chung, quy định đào tạo, FAQ, lịch học, liên hệ hỗ trợ. Đây là use-case thực tế của RAG: xây chatbot hỏi đáp tự động cho chương trình.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | 01_Thong-tin-chung.txt | AI-QA Handbook | 3.365 | source, doc_id |
| 2 | 02_Cau-truc-dao-tao.txt | AI-QA Handbook | 3.298 | source, doc_id |
| 3 | 03_Cong-ty-thuc-tap.txt | AI-QA Handbook | 2.815 | source, doc_id |
| 4 | 04_Thoi-khoa-bieu.txt | AI-QA Handbook | 2.826 | source, doc_id |
| 5 | 05_Danh-gia-qua-trinh.txt | AI-QA Handbook | 2.513 | source, doc_id |
| 6 | 06_He-thong-giang-day.txt | AI-QA Handbook | 3.471 | source, doc_id |
| 7 | 07_Dich-vu-tien-ich.txt | AI-QA Handbook | 4.872 | source, doc_id |
| 8 | 08_Quy-trinh-dao-tao.txt | AI-QA Handbook | 8.758 | source, doc_id |
| 9 | 09_Lien-he-ho-tro.txt | AI-QA Handbook | 7.210 | source, doc_id |
| 10 | FAQ.txt | AI-QA Handbook | 11.542 | source, doc_id |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ | Tại sao hữu ích |
|----------------|------|-------|-----------------|
| source | string | "FAQ.txt", "08_Quy-trinh-dao-tao.txt" | Lọc kết quả theo document nguồn |
| doc_id | string | "FAQ", "08_Quy-trinh-dao-tao" | Group hoặc xóa chunks theo document |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy ChunkingStrategyComparator().compare() trên tài liệu FAQ.txt (11.542 ký tự, chunk_size=200):

| Strategy | Chunk Count | Avg Length | Preserves Context? |
|----------|-------------|------------|-------------------|
| FixedSizeChunker | 58 | 199.0 | Không, cắt giữa câu |
| SentenceChunker | 80 | 143.1 | Có, giữ nguyên câu |
| RecursiveChunker | 85 | 142.9 | Có, tôn trọng ranh giới xuống dòng |

### Strategy Của Tôi

**Loại:** Hybrid Section-Based + Sentence (custom strategy)

**Mô tả cách hoạt động:**

Bước 1: Tách document theo section separators là các dòng "=======" hoặc "---". Mỗi section là một đơn vị ngữ nghĩa hoàn chỉnh, ví dụ một câu hỏi kèm trả lời trong FAQ.

Bước 2: Trong mỗi section, dùng SentenceChunker gom 5 câu thành 1 chunk. Nếu section ngắn dưới 400 ký tự thì giữ nguyên không cắt tiếp.

**Lý do chọn strategy này cho domain:**

Tài liệu AI-QA có cấu trúc section rất rõ ràng. Mỗi câu hỏi FAQ là một section riêng, mỗi chương trong handbook phân cách bằng dấu "=======". Việc split theo section giữ nguyên đơn vị thông tin, mỗi chunk tương đương một câu trả lời hoặc một mục, giúp retrieval chính xác hơn so với cắt mù quáng theo ký tự.

### So Sánh: Strategy của tôi vs Baseline

Kết quả benchmark trên 5 queries, 10 documents:

| Strategy | Precision | Recall | Total Chunks |
|----------|-----------|--------|--------------|
| Full Document (không cắt) | 53% | 90% | 10 |
| Section-Based | 67% | 80% | 150 |
| Sentence-Based (5 câu/chunk) | 60% | 60% | 55 |
| Hybrid Section+Sentence (của tôi) | 67% | 80% | 159 |

Strategy của tôi đạt precision và recall cao nhất trong các strategy có cắt document, ngang bằng Section-Based thuần túy.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**SentenceChunker.chunk:**

Dùng regex re.split(r'(?<=[.!?])\s+|(?<=\.)\n') để tách câu. Pattern này nhìn đằng sau là dấu kết thúc câu (. ! ?) theo sau là khoảng trắng, hoặc dấu chấm theo sau là xuống dòng. Sau đó gom nhóm max_sentences_per_chunk câu thành 1 chunk, strip whitespace. Edge case text rỗng trả về danh sách rỗng.

**RecursiveChunker.chunk và _split:**

Dùng thuật toán đệ quy. Thử split bằng separator đầu tiên trong danh sách ưu tiên (mặc định: xuống dòng kép, xuống dòng đơn, dấu chấm cách, khoảng trắng, chuỗi rỗng). Nếu phần con nhỏ hơn hoặc bằng chunk_size thì giữ lại. Nếu phần con vẫn lớn hơn chunk_size thì đệ quy với separator tiếp theo. Base case là không còn separator thì trả về nguyên phần đó, hoặc split theo ký tự nếu separator là chuỗi rỗng.

### EmbeddingStore

**add_documents và search:**

add_documents duyệt từng Document, gọi embedding_fn trên content để tạo vector, lưu record gồm id, content, metadata và embedding vào danh sách self._store. Search embed query string, tính dot product với embedding của mọi stored record, sắp xếp giảm dần theo score, trả về top-k kết quả mỗi kết quả có content và score.

**search_with_filter và delete_document:**

search_with_filter lọc self._store theo metadata_filter trước, so khớp từng key-value trong filter, rồi chạy similarity search trên tập đã lọc. Nếu metadata_filter là None thì gọi search bình thường. delete_document dùng list comprehension loại bỏ tất cả records có record id trùng doc_id cần xóa. Trả về True nếu số lượng giảm, False nếu không tìm thấy gì để xóa.

### KnowledgeBaseAgent

**answer:**

Gọi store.search(question, top_k) để lấy top-k chunks liên quan. Ghép chunks thành context có đánh số [1], [2], [3]. Build prompt theo mẫu: yêu cầu trả lời dựa trên context, kèm theo context và câu hỏi. Gọi llm_fn(prompt) và trả về chuỗi kết quả.

### Test Results

```
============================= 42 passed in 3.50s ==============================
```

Số tests pass: 42 trên 42.

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Kết quả thực tế | Đúng? |
|------|-----------|-----------|---------|----------------|-------|
| 1 | "Python là ngôn ngữ lập trình phổ biến" | "Python programming language is popular" | cao | 0.9300 | Đúng |
| 2 | "Máy tính cần RAM 16GB để học AI" | "Cấu hình laptop tối thiểu RAM 16GB" | cao | 0.8741 | Đúng |
| 3 | "Chương trình có 3 giai đoạn đào tạo" | "Món phở Hà Nội rất ngon" | thấp | 0.6596 | Sai, cao hơn dự đoán |
| 4 | "Học viên được trợ cấp 8 triệu/tháng" | "8 triệu VND là mức trợ cấp hàng tháng" | cao | 0.8352 | Đúng |
| 5 | "Liên hệ email AIThucchien" | "Discord là kênh hỗ trợ online" | thấp | 0.8461 | Sai, cao hơn dự đoán |

**Kết quả bất ngờ nhất:**

Cặp 3 và 5 có similarity cao hơn nhiều so với dự đoán (0.66 và 0.85), dù nội dung hoàn toàn khác chủ đề. Điều này cho thấy mock embedder dùng MD5 hash không có khả năng hiểu ngữ nghĩa thật sự. Vector sinh ra từ hash có tính chất ngẫu nhiên, hai text khác nhau vẫn có thể có góc nhỏ do trùng hash pattern. Với embedding thật sự như all-MiniLM-L6-v2, cặp khác chủ đề sẽ có similarity thấp hơn rõ rệt.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Chương trình có hỗ trợ tài chính hay trợ cấp không? | Miễn 100% học phí, trợ cấp 8 triệu đồng mỗi tháng trong 12 tuần |
| 2 | Cấu hình laptop tối thiểu để học AI là gì? | CPU Core i7 thế hệ 8 hoặc Apple M2, RAM 16GB, SSD 256GB |
| 3 | Liên hệ ai khi có vấn đề về học phí hoặc hành chính? | Phạm Quốc Khánh, Chuyên viên Quản lý Đào tạo, email AIThucchien@vinuni.edu.vn, phòng I-114 |
| 4 | Chương trình có bao nhiêu giai đoạn và thời lượng mỗi giai đoạn? | 3 giai đoạn: Nền tảng 3 tuần, Chuyên sâu 3 tuần, Thực chiến 6 tuần, tổng 12 tuần |
| 5 | Học viên được nghỉ tối đa bao nhiêu buổi? | Tối đa 04 buổi trong giai đoạn 1 và 2, không nghỉ 02 buổi liên tiếp trong cùng một tuần |

### Kết Quả Của Tôi

| # | Query | Top-1 Source | Relevant? |
|---|-------|-------------|-----------|
| 1 | Trợ cấp tài chính? | FAQ.txt | Có |
| 2 | Cấu hình laptop? | FAQ.txt | Có |
| 3 | Liên hệ hành chính? | 09_Lien-he-ho-tro.txt | Có |
| 4 | Số giai đoạn? | 02_Cau-truc-dao-tao.txt | Có |
| 5 | Nghỉ tối đa? | 08_Quy-trinh-dao-tao.txt | Có |

Số queries trả về chunk relevant trong top-3: 5 trên 5.

### Benchmark So Sánh 4 Strategies

| Strategy | Precision | Recall | Total Chunks |
|----------|-----------|--------|--------------|
| Full Document (không cắt) | 53% | 90% | 10 |
| Section-Based | 67% | 80% | 150 |
| Sentence-Based (5 câu/chunk) | 60% | 60% | 55 |
| Hybrid Section+Sentence (của tôi) | 67% | 80% | 159 |

---

## 7. What I Learned (5 điểm)

### Failure Analysis

**Failure case:** Query "Cấu hình laptop tối thiểu để học AI" — các strategy Sentence-Based và Hybrid không retrieve được chunk chứa từ khóa "Core i7", "SSD", "256GB" trong top-3, dù các từ khóa này có trong FAQ.txt.

**Nguyên nhân:** Chunk chứa thông tin cấu hình laptop bị cắt nhỏ quá, nằm ở section dài với nhiều mục con. Khi embedding bằng mock embedder, chunk này không có đủ overlap từ vựng với query để được xếp hạng cao.

**Đề xuất cải thiện:** Tăng chunk size cho section chứa thông tin chi tiết, hoặc prepend tiêu đề section vào chunk để embedding capture thêm từ khóa.

### Điều học được

**Từ nhóm:** Query expansion — chạy nhiều biến thể của cùng một câu hỏi rồi gộp và sắp xếp lại kết quả — cải thiện đáng kể precision so với tìm kiếm một query duy nhất.

**Từ nhóm khác qua demo:** Nhóm dùng embedder thật sự all-MiniLM-L6-v2 cho kết quả retrieval khác hẳn mock embedder. Các chunks có ngữ nghĩa tương đồng được retrieve chính xác hơn ngay cả khi không trùng từ khóa với query.

**Nếu làm lại:** Sẽ dùng embedder thật sự thay vì mock để benchmark phản ánh đúng chất lượng retrieval. Thêm metadata chi tiết hơn như section_type, language, topic_tags để hỗ trợ filtering chính xác.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5/5 |
| Document selection | Nhóm | 9/10 |
| Chunking strategy | Nhóm | 13/15 |
| My approach | Cá nhân | 9/10 |
| Similarity predictions | Cá nhân | 4/5 |
| Results | Cá nhân | 9/10 |
| Core implementation (tests) | Cá nhân | 30/30 |
| Demo | Nhóm | 4/5 |
| **Tổng** | | **83/100** |
