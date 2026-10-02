# 匯入此段程式需要的 math。
import math

# 匯入此段程式需要的 os。
import os

# 匯入此段程式需要的 Path。
from pathlib import Path

# 匯入此段程式需要的 torch。
import torch

# 匯入此段程式需要的 transformers。
import transformers

# 匯入此段程式需要的 snapshot_download。
from huggingface_hub import snapshot_download

# 匯入此段程式需要的 AutoModelForCausalLM, AutoTokenizer。
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型與生成設定直接放在此處；不讀取外部設定檔。
SETTINGS = {
    # 評分模型，可使用 Hugging Face 名稱或本機模型資料夾。
    "scoring_model": "Qwen/Qwen2.5-0.5B-Instruct",

    # 抽樣與生成模型；與評分模型相同時共用同一份instance。
    "sampling_model": "Qwen/Qwen2.5-0.5B-Instruct",

    # 模型版本；需要固定實驗版本時可改成 commit。
    "revision": "main",

    # 自動選擇 CUDA 或 CPU，也可以指定 cuda:0。
    "device": "auto",

    # 模型快取路徑，以目前工作目錄為基準。
    "cache_dir": ".cache/models",

    # 設成 True 時只讀取已下載的本機模型。
    "local_files_only": False,

    # 保留原文前幾個 token 作為續寫起點。
    "prefix_tokens": 30,

    # 限制上下文長度，避免長文章消耗過多 GPU 記憶體。
    "max_context_tokens": 4096,

    # 分批計算特徵，降低額外機率矩陣的記憶體需求。
    "feature_chunk_tokens": 128,

    # 是否把每個 token 的特徵寫入結果。
    "save_token_features": True,

    # 續寫抽樣使用的亂數種子。
    "seed": 42,

    # 續寫是否使用隨機抽樣；False 則選擇最高機率 token。
    "do_sample": True,

    # 隨機抽樣的溫度。
    "temperature": 0.8,

    # 保留累積機率達到此值的候選 token。
    "top_p": 0.95,

    # 是否同時產生改寫內容。
    "paraphrase_enabled": True,

    # 改寫預設使用確定性的 greedy decoding。
    "paraphrase_do_sample": False,

    # 改寫最多生成的 token 數量。
    "paraphrase_max_new_tokens": 512,

    # 可直接修改的改寫要求；模型输出仍需另外驗證語義。
    "paraphrase_instruction": (

        # 接續此段文字內容，Python 會合併相鄰字串。
        "請用與原文相同的語言改寫下面的文字。保留原意、事實、專有名詞、數字、否定和語氣，"
        "不新增或省略資訊。自然地改寫名詞、動詞、形容詞、副詞及其他詞性的表達，"
        "可調整句型，但不要為替換每個詞而改變語義。只輸出改寫後的文字，不要解釋。"
    ),
}


# 依執行裝置決定模型載入精度，T4 使用 FP16，CPU 使用 FP32。
def select_device_dtype(device="auto"):

    # 未指定裝置時，依目前 CUDA 可用性自動選擇。
    if device == "auto":

        # 計算並保存 device，供後續步驟使用。
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 計算並保存 device，供後續步驟使用。
    device = torch.device(device)

    # CPU 推論使用 FP32，不沿用模型預設的 BF16。
    if device.type == "cpu":

        # 回傳這個步驟的結果，交給呼叫端使用。
        return device, torch.float32

    # 檢查此條件，再決定是否執行下方處理。
    if device.type != "cuda" or not torch.cuda.is_available():

        # 用明確的錯誤說明停止無法正確執行的操作。
        raise ValueError("請選擇可用的 CUDA 裝置或 CPU")

    # 在指定 GPU 上檢查 BF16 能力，避免誤用其他 GPU 的結果。
    with torch.cuda.device(device):

        # 排除 BF16 模擬支援，避免 T4 被誤判為可原生執行 BF16。
        if torch.cuda.is_bf16_supported(including_emulation=False):

            # 計算並保存 dtype，供後續步驟使用。
            dtype = torch.bfloat16

        # 處理前述條件不成立的情況。
        else:

            # 計算並保存 dtype，供後續步驟使用。
            dtype = torch.float16

    # 回傳這個步驟的結果，交給呼叫端使用。
    return device, dtype

# 集中擴充逐 token 特徵；輸入為自然對數機率，輸出維持位置維度。
def extract_token_features(log_probs):

    # 將對數機率或排序後的 logits 轉換為機率。
    probs = log_probs.exp()

    # 回傳這個步驟的結果，交給呼叫端使用。
    return {

        # 計算完整機率分布的 entropy，單位為 nats。
        "entropy": -(probs * log_probs).sum(-1),

        # 加總前十個 token 的機率；詞彙不足十個時使用全部。
        "top10_mass": probs.topk(min(10, probs.shape[-1]), dim=-1).values.sum(-1),

    # 結束這一段參數或資料結構。
    }

# 彙整各特徵的平均值與母體標準差，供分析與生成流程共用。
def summarize_features(features):

    # 建立存放平均值與標準差的結果字典。
    summary = {}

    # 依序處理這個集合中的每一項資料。
    for name, values in features.items():

        # 計算並保存此特徵的平均值或母體標準差。
        summary[f"{name}_mean"] = float(values.mean())

        # 計算並保存此特徵的平均值或母體標準差。
        summary[f"{name}_std"] = float(values.std(correction=0))

    # 回傳這個步驟的結果，交給呼叫端使用。
    return summary

# 計算曲率與抽樣分布統計。
def analytic_statistics(
    # 傳入此參數或條件，完成目前的函式設定。
    score_logits,

    # 傳入此參數或條件，完成目前的函式設定。
    reference_logits,

    # 傳入此參數或條件，完成目前的函式設定。
    labels,

    # 傳入此參數或條件，完成目前的函式設定。
    chunk_tokens=128,

    # 傳入此參數或條件，完成目前的函式設定。
    save_tokens=True,
):

    # 檢查此條件，再決定是否執行下方處理。
    if score_logits.ndim != 2 or score_logits.shape != reference_logits.shape:

        # 用明確的錯誤說明停止無法正確執行的操作。
        raise ValueError("Expected aligned [positions, vocabulary] logits")

    # 檢查此條件，再決定是否執行下方處理。
    if labels.shape != score_logits.shape[:1] or not labels.numel() or chunk_tokens < 1:

        # 用明確的錯誤說明停止無法正確執行的操作。
        raise ValueError("Expected nonempty aligned labels and positive chunk_tokens")

    # 準備收集每一個特徵分塊的結果。
    pieces = {}

    # 依序處理這個集合中的每一項資料。
    for start in range(0, len(labels), chunk_tokens):

        # 決定目前特徵分塊的結束位置。
        end = start + chunk_tokens

        # 半精度模型的特徵改用 FP32 計算，避免 FP16 機率運算下溢；FP64 測試輸入保留原精度。
        calculation_dtype = torch.float64 if score_logits.dtype == torch.float64 else torch.float32

        # 使用穩定精度計算 scoring model 的 log-softmax。
        logp = score_logits[start:end].to(calculation_dtype).log_softmax(-1)

        # 計算 sampling model 的參考機率分布。
        q = reference_logits[start:end].to(calculation_dtype).softmax(-1)

        # 取出實際 token 在 scoring model 下的對數機率。
        observed = logp.gather(-1, labels[start:end, None]).squeeze(-1)

        # 先減去共同基準，讓均勻分布的 variance 在 FP32 下也能精確等於零。
        centered = logp - logp[:, :1]

        # 在參考分布下計算相對於共同基準的期望值。
        expected_offset = (q * centered).sum(-1)

        # 取得參考抽樣分布下的預期 log-likelihood。
        expected = logp[:, 0] + expected_offset

        # 以中心化二階矩計算參考分布的 variance。
        variance = (q * (centered - expected_offset[:, None]).square()).sum(-1)

        # 整理目前位置的統計值與可擴充特徵。
        values = {

            # 保存實際 token 的 log-likelihood。
            "log_likelihood": observed,

            # 保存參考分布下的預期 log-likelihood。
            "expected_log_likelihood": expected,

            # 保留未標準化的 log-likelihood 差值。
            "curvature": observed - expected,

            # 保留參考分布下的 variance。
            "variance": variance,

            # 保留 variance 的平方根。
            "standard_deviation": variance.sqrt(),

            # 合併集中定義的額外特徵，新增特徵時不必修改摘要流程。
            **extract_token_features(logp),

        # 結束這一段參數或資料結構。
        }

        # 檢查此條件，再決定是否執行下方處理。
        if not all(torch.isfinite(v).all() for v in values.values()):

            # 用明確的錯誤說明停止無法正確執行的操作。
            raise ValueError("Non-finite feature values")

        # 依序處理這個集合中的每一項資料。
        for name, value in values.items():

            # 依照原始順序加入這一項結果。
            pieces.setdefault(name, []).append(value.detach().cpu().double())

    # 準備或整理每個特徵在所有位置上的數值。
    features = {}

    # 依序處理這個集合中的每一項資料。
    for name, parts in pieces.items():

        # 計算並保存 features[name]，供後續步驟使用。
        features[name] = torch.cat(parts)

    # 加總各固定前綴下的 variance，得到 log-likelihood 總和的 variance。
    total_variance = float(features["variance"].sum())

    # 取總 variance 的平方根，作為標準化曲率的分母。
    std = math.sqrt(total_variance)

    # 加總實際與預期 log-likelihood 的差值，得到未標準化曲率。
    curvature = float(features["curvature"].sum())

    # 建立或取得這個階段的完整結果。
    result = {

        # 記錄真正參與評分的位置數，不包含僅作前綴的第一個 token。
        "scored_token_count": len(labels),

        # 保存實際 token 的 log-likelihood。
        "log_likelihood": float(features["log_likelihood"].sum()),

        # 保存參考分布下的預期 log-likelihood。
        "expected_log_likelihood": float(features["expected_log_likelihood"].sum()),

        # 保留未標準化的 log-likelihood 差值。
        "curvature": curvature,

        # 保留參考分布下的 variance。
        "variance": total_variance,

        # 保留 variance 的平方根。
        "standard_deviation": std,

        # 以標準差正規化曲率；零 variance 時回傳 None。
        "discrepancy_score": curvature / std if std > 1e-12 else None,

        # 明確區分正常評分、零 variance 與沒有足夠 token。
        "discrepancy_status": "ok" if std > 1e-12 else "zero_variance",

        # 保存 features 欄位，供後續分析或重現使用。
        "features": summarize_features(features),

    # 結束這一段參數或資料結構。
    }

    # 檢查此條件，再決定是否執行下方處理。
    if save_tokens:

        # 將 "token_features" 寫入這個階段的結果。
        result["token_features"] = {name: value.tolist() for name, value in features.items()}

    # 回傳這個步驟的結果，交給呼叫端使用。
    return result

# 建立可重用的模型介面。
class TokenInfo:

    # 初始化設定與模型，讓同一個模型只載入一次。
    def __init__(self, settings=None):

        # 每個instance複製設定，之後修改全域 SETTINGS 不會改動已載入模型的參數。
        self.settings = SETTINGS.copy()

        # 檢查此條件，再決定是否執行下方處理。
        if settings is not None:

            # 找出未定義的設定鍵，避免拼字錯誤被忽略。
            unknown = settings.keys() - SETTINGS.keys()

            # 檢查此條件，再決定是否執行下方處理。
            if unknown:

                # 用明確的錯誤說明停止無法正確執行的操作。
                raise ValueError(f"未知設定：{sorted(unknown)}")

            # 執行此步驟，更新或驗證目前的處理狀態。
            self.settings.update(settings)

        # 使用目前模型instance的設定快照。
        settings = self.settings

        # 列出所有必須為正整數的長度設定。
        lengths = ["prefix_tokens", "max_context_tokens", "feature_chunk_tokens", "paraphrase_max_new_tokens"]

        # 依序處理這個集合中的每一項資料。
        for name in lengths:

            # 檢查此條件，再決定是否執行下方處理。
            if type(settings[name]) is not int or settings[name] < 1:

                # 用明確的錯誤說明停止無法正確執行的操作。
                raise ValueError(f"{name} 必須是正整數")

        # 檢查此條件，再決定是否執行下方處理。
        if not math.isfinite(settings["temperature"]) or settings["temperature"] <= 0:

            # 用明確的錯誤說明停止無法正確執行的操作。
            raise ValueError("temperature 必須是有限的正數")

        # 檢查此條件，再決定是否執行下方處理。
        if not 0 < settings["top_p"] <= 1:

            # 用明確的錯誤說明停止無法正確執行的操作。
            raise ValueError("top_p 必須大於 0 且不超過 1")

        # 偵測裝置並決定真正用於載入模型的精度。
        self.device, self.dtype = select_device_dtype(settings["device"])

        # 建立已載入模型的共用紀錄。
        self._models = {}

        # 取得 scoring model 與對應 tokenizer。
        self.scoring_tokenizer, self.scoring_model = self._load(settings["scoring_model"])

        # 取得 sampling model；名稱相同時會使用既有instance。
        self.sampling_tokenizer, self.sampling_model = self._load(settings["sampling_model"])

        # 檢查此條件，再決定是否執行下方處理。
        if self.scoring_tokenizer.get_vocab() != self.sampling_tokenizer.get_vocab():

            # 用明確的錯誤說明停止無法正確執行的操作。
            raise ValueError("Scoring/sampling models need identical token-to-ID vocabularies for analytic discrepancy")

        # 保存有效詞彙數，排除模型輸出的 padding logits。
        self.vocab_size = len(self.scoring_tokenizer)

        # 檢查此條件，再決定是否執行下方處理。
        if set(self.scoring_tokenizer.get_vocab().values()) != set(range(self.vocab_size)):

            # 用明確的錯誤說明停止無法正確執行的操作。
            raise ValueError("Tokenizer must have contiguous token IDs")

        # 建立此instance專用的抽樣亂數產生器。
        self.rng = torch.Generator(device=self.device).manual_seed(settings["seed"])

    # 載入尚未建立的模型與 tokenizer，重複要求時直接回傳既有instance。
    def _load(self, name):

        # 正規化本機模型路徑，作為模型共用的查找鍵。
        key = os.path.normcase(str(Path(name).resolve())) if Path(name).is_dir() else name

        # 檢查此條件，再決定是否執行下方處理。
        if key not in self._models:

            # 集中本次模型與 tokenizer 載入時共用的選項。
            options = {

                # 保存 cache_dir 欄位，供後續分析或重現使用。
                "cache_dir": self.settings["cache_dir"],

                # 保存 revision 欄位，供後續分析或重現使用。
                "revision": self.settings["revision"],

                # 保存 local_files_only 欄位，供後續分析或重現使用。
                "local_files_only": self.settings["local_files_only"],

                # 保存 trust_remote_code 欄位，供後續分析或重現使用。
                "trust_remote_code": False,

            # 結束這一段參數或資料結構。
            }

            # 離線時先解析快取目錄，避免 tokenizer 額外向 Hub 查詢。
            source = name

            # 檢查此條件，再決定是否執行下方處理。
            if self.settings["local_files_only"] and not Path(name).is_dir():

                # 決定實際模型來源，可以是模型名稱或本機快取目錄。
                source = snapshot_download(

                    # 傳入此參數或條件，完成目前的函式設定。
                    name,

                    # 傳入此參數或條件，完成目前的函式設定。
                    cache_dir=self.settings["cache_dir"],

                    # 傳入此參數或條件，完成目前的函式設定。
                    revision=self.settings["revision"],

                    # 傳入此參數或條件，完成目前的函式設定。
                    local_files_only=True,

                # 結束這一段參數或資料結構。
                )

            # 取得這次處理所使用的 tokenizer。
            tokenizer = AutoTokenizer.from_pretrained(source, **options)

            # 依偵測到的 dtype 載入模型，避免沿用模型設定中的不相容精度。
            model = AutoModelForCausalLM.from_pretrained(

                # 傳入此參數或條件，完成目前的函式設定。
                source,

                # 傳入此參數或條件，完成目前的函式設定。
                dtype=self.dtype,

                # 套用本次模型載入所需的共用參數。
                **options,

            # 結束這一段參數或資料結構。
            ).to(self.device).eval()

            # 檢查此條件，再決定是否執行下方處理。
            if Path(source).parent.name == "snapshots":

                # 計算並保存 model.config._commit_hash，供後續步驟使用。
                model.config._commit_hash = Path(source).name

            # 保存 self._models[key]，供這個instance或測試類別後續使用。
            self._models[key] = (tokenizer, model)

        # 回傳這個步驟的結果，交給呼叫端使用。
        return self._models[key]

    # 同時遵守模型與使用者的上下文上限，不靜默截斷文章。
    def _check_length(self, length, model):

        # 讀取模型本身可接受的最長上下文。
        model_limit = getattr(model.config, "max_position_embeddings", self.settings["max_context_tokens"])

        # 取使用者設定與模型上限中較小的值。
        limit = min(self.settings["max_context_tokens"], model_limit)

        # 檢查此條件，再決定是否執行下方處理。
        if length > limit:

            # 用明確的錯誤說明停止無法正確執行的操作。
            raise ValueError(f"{length} tokens exceed context limit {limit}; text was not truncated")

    # 將文字轉成 token IDs，並檢查兩個 tokenizer 的切詞一致性。
    def encode(self, text):

        # 檢查此條件，再決定是否執行下方處理。
        if not isinstance(text, str) or not text.strip():

            # 用明確的錯誤說明停止無法正確執行的操作。
            raise ValueError("Input must be a nonempty string")

        # 準備文字或生成上下文的 token ID 序列。
        ids = self.scoring_tokenizer.encode(text, add_special_tokens=False)

        # 檢查此條件，再決定是否執行下方處理。
        if ids != self.sampling_tokenizer.encode(text, add_special_tokens=False):

            # 用明確的錯誤說明停止無法正確執行的操作。
            raise ValueError("Tokenizers segment this text differently")

        # 檢查此條件，再決定是否執行下方處理。
        if not ids:

            # 用明確的錯誤說明停止無法正確執行的操作。
            raise ValueError("Input produced no tokens")

        # 回傳這個步驟的結果，交給呼叫端使用。
        return ids

    # 關閉梯度追蹤，降低推論時的記憶體消耗。
    @torch.inference_mode()

    # 用已知 token 序列評分；第一個 token 只提供上下文。
    def analyze_ids(self, ids):

        # 依序處理這個集合中的每一項資料。
        for model in (self.scoring_model, self.sampling_model):

            # 執行此步驟，更新或驗證目前的處理狀態。
            self._check_length(len(ids), model)

        # 檢查此條件，再決定是否執行下方處理。
        if len(ids) < 2:

            # 回傳這個步驟的結果，交給呼叫端使用。
            return {

                # 記錄真正參與評分的位置數，不包含僅作前綴的第一個 token。
                "scored_token_count": 0,

                # 以標準差正規化曲率；零 variance 時回傳 None。
                "discrepancy_score": None,

                # 明確區分正常評分、零 variance 與沒有足夠 token。
                "discrepancy_status": "insufficient_tokens",

            # 結束這一段參數或資料結構。
            }

        # 把整段 token IDs 放到模型所在裝置。
        tokens = torch.tensor([ids], device=self.device)

        # 取得 scoring logits，將位置向左對齊下一個實際 token。
        score = self.scoring_model(input_ids=tokens, use_cache=False).logits[0, :-1, :self.vocab_size]

        # 同模型不重複 forward，直接共用 logits。
        if self.scoring_model is self.sampling_model:

            # 取得對齊的參考 logits；同模型時直接重用 scoring 結果。
            reference = score

        # 處理前述條件不成立的情況。
        else:

            # 取得對齊的參考 logits；同模型時直接重用 scoring 結果。
            reference = self.sampling_model(input_ids=tokens, use_cache=False).logits[0, :-1, :self.vocab_size]

        # 建立或取得這個階段的完整結果。
        result = analytic_statistics(

            # 傳入此參數或條件，完成目前的函式設定。
            score,

            # 傳入此參數或條件，完成目前的函式設定。
            reference,

            # 傳入此參數或條件，完成目前的函式設定。
            tokens[0, 1:],

            # 傳入此參數或條件，完成目前的函式設定。
            self.settings["feature_chunk_tokens"],

            # 傳入此參數或條件，完成目前的函式設定。
            self.settings["save_token_features"],

        # 結束這一段參數或資料結構。
        )

        # 依設定決定是否保留每個 token 的詳細特徵。
        if self.settings["save_token_features"]:

            # 將 "token_features"]["token_id" 寫入這個階段的結果。
            result["token_features"]["token_id"] = ids[1:]

            # 將 "token_features"]["position" 寫入這個階段的結果。
            result["token_features"]["position"] = list(range(1, len(ids)))

        # 回傳這個步驟的結果，交給呼叫端使用。
        return result

    # 提供可直接輸入文字的分析介面。
    def analyze(self, text):

        # 回傳這個步驟的結果，交給呼叫端使用。
        return self.analyze_ids(self.encode(text))

    # 關閉梯度追蹤，降低推論時的記憶體消耗。
    @torch.inference_mode()

    # 以 KV cache 逐步生成，並在每一步擷取尚未套用抽樣限制的特徵。
    def _generate(self, prompt_ids, count, exact, on_step=None):

        # 執行此步驟，更新或驗證目前的處理狀態。
        self._check_length(len(prompt_ids) + count, self.sampling_model)

        # 準備文字或生成上下文的 token ID 序列。
        ids = list(prompt_ids)

        # 保留已計算的 KV cache，避免重算整段前綴。
        cache = None

        # 準備存放每個新增 token 的原始分布特徵。
        rows = []

        # 準備存放使用者 on_step 回傳的額外特徵。
        custom_rows = []

        # 整理模型允許的結束 token IDs。
        eos = self.sampling_model.generation_config.eos_token_id

        # 檢查此條件，再決定是否執行下方處理。
        if eos is None:

            # 整理模型允許的結束 token IDs。
            eos = set()

        # 檢查此條件，再決定是否執行下方處理。
        elif isinstance(eos, int):

            # 整理模型允許的結束 token IDs。
            eos = {eos}

        # 處理前述條件不成立的情況。
        else:

            # 整理模型允許的結束 token IDs。
            eos = set(eos)

        # 檢查此條件，再決定是否執行下方處理。
        if self.sampling_tokenizer.eos_token_id is not None:

            # 執行此步驟，更新或驗證目前的處理狀態。
            eos.add(self.sampling_tokenizer.eos_token_id)

        # 記錄生成停止的原因。
        reason = "length"

        # 依序處理這個集合中的每一項資料。
        for step in range(count):

            # 第一次傳入完整提示詞，之後只傳入最後一個 token。
            inputs = ids if cache is None else ids[-1:]

            # 執行模型預測，或指定本次輸出資料的檔案位置。
            output = self.sampling_model(

                # 傳入此參數或條件，完成目前的函式設定。
                input_ids=torch.tensor([inputs], device=self.device),

                # 傳入此參數或條件，完成目前的函式設定。
                attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=self.device),

                # 傳入此參數或條件，完成目前的函式設定。
                past_key_values=cache,

                # 傳入此參數或條件，完成目前的函式設定。
                use_cache=True,

            # 結束這一段參數或資料結構。
            )

            # 保留已計算的 KV cache，避免重算整段前綴。
            cache = output.past_key_values

            # 取出最後位置的原始 logits 並轉為 FP32，不先改變抽樣分布。
            raw = output.logits[0, -1, :self.vocab_size].float()

            # 建立用於抽樣的 logits，保留原始數值供特徵抽取。
            logits = raw.clone()

            # 取得需要禁止生成的特殊 token IDs。
            blocked = set(self.sampling_tokenizer.all_special_ids)

            # 檢查此條件，再決定是否執行下方處理。
            if not exact:

                # 改寫允許生成 EOS，讓模型可以正常停止。
                blocked -= eos

            # 檢查此條件，再決定是否執行下方處理。
            if blocked:

                # 計算並保存 logits[list(blocked)]，供後續步驟使用。
                logits[list(blocked)] = -torch.inf

            # 分別套用續寫或改寫的隨機抽樣開關。
            do_sample = self.settings["do_sample"] if exact else self.settings["paraphrase_do_sample"]

            # 啟用隨機抽樣時，先套用 temperature 與 top-p。
            if do_sample:

                # 依設定溫度調整抽樣分布。
                logits /= self.settings["temperature"]

                # 依 logits 由高到低排序，準備進行 top-p 篩選。
                sorted_logits, order = logits.sort(descending=True)

                # 將對數機率或排序後的 logits 轉換為機率。
                probs = sorted_logits.softmax(-1)

                # 標記已超過 top-p 範圍的候選 token，保留跨過門檻的那一項。
                remove = probs.cumsum(-1) - probs > self.settings["top_p"]

                # 計算並保存 sorted_logits[remove]，供後續步驟使用。
                sorted_logits[remove] = -torch.inf

                # 用專屬亂數產生器從篩選後的分布抽樣。
                sampled_index = torch.multinomial(sorted_logits.softmax(-1), 1, generator=self.rng)

                # 取得這一步選中的實際 token ID。
                chosen = int(order[sampled_index])

            # 處理前述條件不成立的情況。
            else:

                # 取得這一步選中的實際 token ID。
                chosen = int(logits.argmax())

            # 檢查此條件，再決定是否執行下方處理。
            if not torch.isfinite(logits[chosen]):

                # 用明確的錯誤說明停止無法正確執行的操作。
                raise ValueError("No valid token available for generation")

            # 改寫遇到 EOS 即停止，不把 EOS 計入文字特徵。
            if not exact and chosen in eos:

                # 記錄生成停止的原因。
                reason = "eos"

                # 停止目前的逐 token 生成迴圈。
                break

            # 整理目前位置的統計值與可擴充特徵。
            values = extract_token_features(raw.log_softmax(-1))

            # 依照原始順序加入這一項結果。
            rows.append({name: float(value) for name, value in values.items()})

            # 只有使用者提供擴充 callback 時才執行額外特徵處理。
            if on_step is not None:

                # 依照原始順序加入這一項結果。
                custom_rows.append(on_step(tuple(ids), raw, chosen))

            # 依照原始順序加入這一項結果。
            ids.append(chosen)

        # 準備或整理每個特徵在所有位置上的數值。
        features = {}

        # 檢查此條件，再決定是否執行下方處理。
        if rows:

            # 依序處理這個集合中的每一項資料。
            for name in rows[0]:

                # 計算並保存 features[name]，供後續步驟使用。
                features[name] = torch.tensor([row[name] for row in rows], dtype=torch.float64)

        # 整理生成過程的摘要、分布來源與停止原因。
        info = {

            # 保存 stop_reason 欄位，供後續分析或重現使用。
            "stop_reason": reason,

            # 保存 features 欄位，供後續分析或重現使用。
            "features": summarize_features(features),

            # 保存 feature_model 欄位，供後續分析或重現使用。
            "feature_model": self.settings["sampling_model"],

            # 保存 distribution 欄位，供後續分析或重現使用。
            "distribution": "raw_before_sampling_filters",

        # 結束這一段參數或資料結構。
        }

        # 依設定決定是否保留每個 token 的詳細特徵。
        if self.settings["save_token_features"]:

            # 計算並保存 info["token_features"]，供後續步驟使用。
            info["token_features"] = rows

        # 只有使用者提供擴充 callback 時才執行額外特徵處理。
        if on_step is not None:

            # 計算並保存 info["custom_token_features"]，供後續步驟使用。
            info["custom_token_features"] = custom_rows

        # 回傳這個步驟的結果，交給呼叫端使用。
        return ids[len(prompt_ids):], info

    # 將文字、token 序列與分析結果組成共同資料結構。
    def _record(self, ids, text=None):

        # 檢查此條件，再決定是否執行下方處理。
        if text is None:

            # 取得 token 序列的完整解碼文字。
            text = self.scoring_tokenizer.decode(ids, clean_up_tokenization_spaces=False)

        # 回傳這個步驟的結果，交給呼叫端使用。
        return {

            # 保存 text 欄位，供後續分析或重現使用。
            "text": text,

            # 保存 token_ids 欄位，供後續分析或重現使用。
            "token_ids": ids,

            # 保存 token_count 欄位，供後續分析或重現使用。
            "token_count": len(ids),

            # 保存 analysis 欄位，供後續分析或重現使用。
            "analysis": self.analyze_ids(ids),

        # 結束這一段參數或資料結構。
        }

    # 保留原文 prefix，生成與原文 token 數完全相同的完整續寫。
    def continue_text(self, text, on_step=None):

        # 保留原文的完整 token 序列。
        original = self.encode(text)

        # 依設定取出原文前幾個 token，短文章則使用全部內容。
        prefix = original[:self.settings["prefix_tokens"]]

        # 取得新生成的 token IDs 與生成過程特徵。
        generated, generation = self._generate(prefix, len(original) - len(prefix), True, on_step)

        # 直接串接 prefix 與新增 token IDs，維持精確的長度定義。
        full = prefix + generated

        # 檢查此條件，再決定是否執行下方處理。
        if len(full) != len(original):

            # 用明確的錯誤說明停止無法正確執行的操作。
            raise RuntimeError("Continuation length invariant failed")

        # 建立或取得這個階段的完整結果。
        result = self._record(full)

        # 將 "prefix_token_ids" 寫入這個階段的結果。
        result["prefix_token_ids"] = prefix

        # 將 "prefix" 寫入這個階段的結果。
        result["prefix"] = self.scoring_tokenizer.decode(prefix, clean_up_tokenization_spaces=False)

        # 將 "continuation_token_ids" 寫入這個階段的結果。
        result["continuation_token_ids"] = generated

        # 將 "continuation" 寫入這個階段的結果。
        result["continuation"] = self.scoring_tokenizer.decode(generated, clean_up_tokenization_spaces=False)

        # 將 "generation" 寫入這個階段的結果。
        result["generation"] = generation

        # 另行計算解碼後重新切詞的結果，不以它取代原始 token IDs。
        retokenized = self.scoring_tokenizer.encode(result["text"], add_special_tokens=False)

        # 將 "retokenized_token_count" 寫入這個階段的結果。
        result["retokenized_token_count"] = len(retokenized)

        # 回傳這個步驟的結果，交給呼叫端使用。
        return result

    # 利用原生對話提示詞改寫文字，保留語義待驗證的標記。
    def paraphrase(self, text, on_step=None):

        # 執行此步驟，更新或驗證目前的處理狀態。
        self.encode(text)

        # 組合改寫要求與原文，供沒有 chat template 的模型使用。
        prompt = self.settings["paraphrase_instruction"] + "\n\n" + text

        # 取得這次處理所使用的 tokenizer。
        tokenizer = self.sampling_tokenizer

        # 檢查此條件，再決定是否執行下方處理。
        if tokenizer.chat_template:

            # 把改寫要求放入 system 訊息，原文放入 user 訊息。
            messages = [

                # 傳入此參數或條件，完成目前的函式設定。
                {"role": "system", "content": self.settings["paraphrase_instruction"]},

                # 傳入此參數或條件，完成目前的函式設定。
                {"role": "user", "content": text},

            # 結束這一段參數或資料結構。
            ]

            # 建立改寫提示詞的 token IDs。
            prompt_ids = tokenizer.apply_chat_template(

                # 傳入此參數或條件，完成目前的函式設定。
                messages,

                # 傳入此參數或條件，完成目前的函式設定。
                add_generation_prompt=True,

                # 傳入此參數或條件，完成目前的函式設定。
                tokenize=True,

                # 傳入此參數或條件，完成目前的函式設定。
                return_dict=False,

            # 結束這一段參數或資料結構。
            )

        # 處理前述條件不成立的情況。
        else:

            # 建立改寫提示詞的 token IDs。
            prompt_ids = tokenizer.encode(prompt + "\n\n", add_special_tokens=False)

        # 取得新生成的 token IDs 與生成過程特徵。
        generated, generation = self._generate(

            # 傳入此參數或條件，完成目前的函式設定。
            prompt_ids,

            # 傳入此參數或條件，完成目前的函式設定。
            self.settings["paraphrase_max_new_tokens"],

            # 傳入此參數或條件，完成目前的函式設定。
            False,

            # 傳入此參數或條件，完成目前的函式設定。
            on_step,

        # 結束這一段參數或資料結構。
        )

        # 取得 token 序列的完整解碼文字。
        text = tokenizer.decode(generated, clean_up_tokenization_spaces=False)

        # 檢查此條件，再決定是否執行下方處理。
        if not generated or not text.strip():

            # 用明確的錯誤說明停止無法正確執行的操作。
            raise ValueError("Model produced an empty paraphrase")

        # 建立或取得這個階段的完整結果。
        result = self._record(generated, text)

        # 將 "generation" 寫入這個階段的結果。
        result["generation"] = generation

        # 將 "prompt_token_ids" 寫入這個階段的結果。
        result["prompt_token_ids"] = prompt_ids

        # 將 "semantic_validation" 寫入這個階段的結果。
        result["semantic_validation"] = "not_verified"

        # 將 "possibly_truncated" 寫入這個階段的結果。
        result["possibly_truncated"] = generation["stop_reason"] == "length"

        # 回傳這個步驟的結果，交給呼叫端使用。
        return result

    # 依序分析原文、續寫與改寫，每個階段各自保留結果或錯誤。
    def process(self, text):

        # 準備文字或生成上下文的 token ID 序列。
        ids = self.encode(text)

        # 建立或取得這個階段的完整結果。
        result = {

            # 保存 original 欄位，供後續分析或重現使用。
            "original": {"text": text, "token_ids": ids, "token_count": len(ids)},

            # 保存 continued 欄位，供後續分析或重現使用。
            "continued": None,

            # 保存 paraphrased 欄位，供後續分析或重現使用。
            "paraphrased": None,

            # 保存 errors 欄位，供後續分析或重現使用。
            "errors": {},

        # 結束這一段參數或資料結構。
        }

        # 每個階段直接執行，不建立額外的函式派送表；失敗時仍保留其他成果。
        try:

            # 將 "original" 寫入這個階段的結果。
            result["original"] = self._record(ids, text)

        # 記錄此階段錯誤，讓其他已完成資料仍能保留。
        except (ValueError, RuntimeError) as error:

            # 將 "errors"]["original" 寫入這個階段的結果。
            result["errors"]["original"] = f"{type(error).__name__}: {error}"

        # 執行此階段，並保留可單獨回報的錯誤範圍。
        try:

            # 將 "continued" 寫入這個階段的結果。
            result["continued"] = self.continue_text(text)

        # 記錄此階段錯誤，讓其他已完成資料仍能保留。
        except (ValueError, RuntimeError) as error:

            # 將 "errors"]["continued" 寫入這個階段的結果。
            result["errors"]["continued"] = f"{type(error).__name__}: {error}"

        # 檢查此條件，再決定是否執行下方處理。
        if self.settings["paraphrase_enabled"]:

            # 執行此階段，並保留可單獨回報的錯誤範圍。
            try:

                # 將 "paraphrased" 寫入這個階段的結果。
                result["paraphrased"] = self.paraphrase(text)

            # 記錄此階段錯誤，讓其他已完成資料仍能保留。
            except (ValueError, RuntimeError) as error:

                # 將 "errors"]["paraphrased" 寫入這個階段的結果。
                result["errors"]["paraphrased"] = f"{type(error).__name__}: {error}"

        # 回傳這個步驟的結果，交給呼叫端使用。
        return result

    # 記錄本次設定、執行裝置、套件版本與模型來源。
    def metadata(self):

        # 準備記錄每個已載入模型的來源與精度。
        models = {}

        # 依序處理這個集合中的每一項資料。
        for name, (_, model) in self._models.items():

            # 計算並保存 models[name]，供後續步驟使用。
            models[name] = {

                # 保存 commit 欄位，供後續分析或重現使用。
                "commit": getattr(model.config, "_commit_hash", None),

                # 保存 dtype 欄位，供後續分析或重現使用。
                "dtype": str(model.dtype),

            # 結束這一段參數或資料結構。
            }

        # 回傳這個步驟的結果，交給呼叫端使用。
        return {

            # 保存 settings 欄位，供後續分析或重現使用。
            "settings": self.settings.copy(),

            # 保存 device 欄位，供後續分析或重現使用。
            "device": str(self.device),

            # 保存 dtype 欄位，供後續分析或重現使用。
            "dtype": str(self.dtype),

            # 保存 torch_version 欄位，供後續分析或重現使用。
            "torch_version": torch.__version__,

            # 保存 transformers_version 欄位，供後續分析或重現使用。
            "transformers_version": transformers.__version__,

            # 保存 loaded_model_count 欄位，供後續分析或重現使用。
            "loaded_model_count": len(self._models),

            # 保存 models 欄位，供後續分析或重現使用。
            "models": models,

        # 結束這一段參數或資料結構。
        }
