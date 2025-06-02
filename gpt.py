import tiktoken
from forex_python.converter import CurrencyRates


def dollar_to_won(d):
    # 요율 (단위: 달러/토큰)
    c = CurrencyRates()
    rate = c.get_rate('USD', 'KRW')
    return d*rate

def calculate_gpt4o_mini__input_cost(prompt_tokens: int):
    input_rate = 0.15 / 1_000_000
    return prompt_tokens * input_rate

def calculate_gpt4o_mini__output_cost(output_tokens: int):
    output_rate = 0.60 / 1_000_000
    return output_tokens * output_rate

def calculate_gpt4o_mini_cost(prompt: str, output: str) -> float:
    # GPT-4o는 cl100k_base 토크나이저 사용
    enc = tiktoken.get_encoding("cl100k_base")
    
    prompt_tokens = len(enc.encode(prompt))
    output_tokens = len(enc.encode(output))
    
    # 요금 계산
    input_cost = calculate_gpt4o_mini__input_cost(prompt_tokens)
    output_cost = calculate_gpt4o_mini__output_cost(output_tokens)
    total_cost = input_cost + output_cost

    return total_cost


if __name__ == "__main__":
    input_tokens = 120791

    input_cost = dollar_to_won(calculate_gpt4o_mini__input_cost(input_tokens))
    print(f"{input_cost:,.0f}원")

    
    


    