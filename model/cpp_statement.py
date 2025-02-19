import re
def is_for_statement(statement):
    if 'for' in statement:
        return True
    return False

def is_while_statement(statement):
    if 'while' in statement:
        return True
    return False
def is_do_statement(statement):
    if 'do' in statement:
        return True
    return False

def is_if_statement(statement):
    if 'if' in statement:
        return True
    return False

def is_else_statement(statement):
    if 'else' in statement:
        return True
    return False

def is_switch_statement(statement):
    if 'switch' in statement:
        return True
    return False

def is_case_statement(statement):
    if 'case' in statement:
        return True
    return False

def is_try_statement(statement):
    if 'try' in statement:
        return True
    return False

def is_catch_statement(statement):
    if 'catch' in statement:
        return True
    return False

def is_throw_statement(statement):
    if 'throw' in statement:
        return True
    return False

def is_return_statement(statement):
    if 'return' in statement:
        return True
    return False

def is_struct_declaration(statement):
    if 'struct' in statement:
        return True
    return False

def is_class_declaration(statement):
    if 'class' in statement:
        return True
    return False

def is_variable(statement):
    if '=' in statement:
        return True
    return False

def is_function_caller(statement):
    # 定义正则表达式，匹配函数调用
    function_call_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*\s*(::\s*[a-zA-Z_][a-zA-Z0-9_]*)?\s*\(.*\)\s*;?$'
    if re.match(function_call_pattern, statement.strip()):
        return True
    return False

def is_pointer(statement):
    # 定义正则表达式，匹配指针操作
    pointer_patterns = [
        r'\*\s*[a-zA-Z_][a-zA-Z0-9_]*',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*&\s*[a-zA-Z_][a-zA-Z0-9_]*',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*\+\+',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*--',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*->\s*[a-zA-Z_][a-zA-Z0-9_]*',
        r'\[\s*\]',
        r'new\s+[a-zA-Z_][a-zA-Z0-9_]*',
        r'\bm?alloc\s*\(.*\)',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*==\s*nullptr',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*!=\s*nullptr'
    ]
    combined_pointer_pattern = '|'.join(pointer_patterns)
    if re.match(combined_pointer_pattern, statement.strip()):
        return True
    return False

def is_expression(statement):
    # 定义正则表达式，匹配常见的表达式形式
    expression_patterns = [
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*.+;',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*[\+\-\*/%]=\s*.+;',
        r'.+\s*[\+\-\*/%]\s*.+;',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*\(.+\);',
        r'.+\s*[\&\|\^]=\s*.+;',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*[\+\-]{2};',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*\?\s*.+:\s*.+;',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*[\<\>\!\=]=\s*.+;',
        r'.+\s*[\&\|\^]{1,2}\s*.+;',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*(\.\s*|\->\s*).+;'
    ]
    combined_expression_pattern = '|'.join(expression_patterns)
    if re.match(combined_expression_pattern, statement.strip()):
        return True
    return False

