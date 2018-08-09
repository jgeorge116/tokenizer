import ply.lex as lex
import ply.yacc as yacc
import sys

'''
Joel George
110148892
'''


class Node:
    def __init__(self):
        print("init node")

    def evaluate(self):
        return 0

    def execute(self):
        return 0


class NumberNode(Node):
    def __init__(self, v):
        if '.' in v:
            self.value = float(v)
        else:
            self.value = int(v)

    def evaluate(self):
        return self.value


class BopNode(Node):
    def __init__(self, op, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.op = op

    def evaluate(self):
        if self.op == '+':
            return self.v1.evaluate() + self.v2.evaluate()
        elif self.op == '-':
            return self.v1.evaluate() - self.v2.evaluate()
        elif self.op == '*':
            return self.v1.evaluate() * self.v2.evaluate()
        elif self.op == '/':
            return self.v1.evaluate() / self.v2.evaluate()
        elif self.op == '**':
            return self.v1.evaluate() ** self.v2.evaluate()
        elif self.op == '%':
            return self.v1.evaluate() % self.v2.evaluate()
        elif self.op == '//':
            return self.v1.evaluate() // self.v2.evaluate()
        elif self.op == '<>':
            return self.v1.evaluate() != self.v2.evaluate()
        elif self.op == '<':
            return self.v1.evaluate() < self.v2.evaluate()
        elif self.op == '<=':
            return self.v1.evaluate() <= self.v2.evaluate()
        elif self.op == '>':
            return self.v1.evaluate() > self.v2.evaluate()
        elif self.op == '>=':
            return self.v1.evaluate() >= self.v2.evaluate()
        elif self.op == '==':
            return self.v1.evaluate() == self.v2.evaluate()
        elif self.op == 'in':
            return self.v1.evaluate() in self.v2.evaluate()


class BooleanNode(Node):
    def __init__(self, op, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.op = op

    def evaluate(self):
        if self.op == 'not':
            return not self.v1.evaluate()
        elif self.op == 'and':
            return self.v1.evaluate() and self.v2.evaluate()
        elif self.op == 'or':
            return self.v1.evaluate() or self.v2.evaluate()
        else:
            return self.v1


class PrintNode(Node):
    def __init__(self, v):
        self.value = v

    def execute(self):
        print(self.value.evaluate())


class BlockNode(Node):
    def __init__(self, s):
        self.sl = [s]

    def execute(self):
        for statement in self.sl:
            statement.execute()


class ListNode(Node):
    def __init__(self, s):
        if s is not None:
            self.s1 = [s]
        else:
            self.s1 = []

    def evaluate(self):
        li = []
        for i in range(len(self.s1)):
            li.append(self.s1[i].evaluate())
        return li


class IndexNode(Node):
    def __init__(self, s, ind):
        self.s1 = s
        self.index = ind

    def evaluate(self):
        return self.s1.evaluate()[self.index.evaluate()]


class StringNode(Node):
    def __init__(self, v):
        self.value = v

    def evaluate(self):
        return self.value


class VariableNode(Node):
    def __init__(self, v):
        self.name = v

    def evaluate(self):
        return variables.get(self.name)


class AssignNode(Node):
    def __init__(self, n, val, ind):
        self.name = n
        self.value = val
        self.index = ind

    def execute(self):
        if self.index is None:
            variables[self.name] = self.value.evaluate()
        else:
            self.name.evaluate()[self.index.evaluate()] = self.value.evaluate()


class IfNode(Node):
    def __init__(self, cond, b):
        self.condition = cond
        self.block = b

    def execute(self):
        if self.condition.evaluate():
            self.block.execute()


class IfElseNode(Node):
    def __init__(self, n, m):
        self.ifNode = n
        self.elseNode = m

    def execute(self):
        if self.ifNode.condition.evaluate():
            self.ifNode.block.execute()
        else:
            self.elseNode.execute()


class WhileNode(Node):
    def __init__(self, cond, b):
        self.condition = cond
        self.block = b

    def execute(self):
        while self.condition.evaluate():
            self.block.execute()


reserved = {
    'print': 'PRINT',
    'if': 'IF',
    'else': 'ELSE',
    'while': 'WHILE',
    'or': 'OR',
    'and': 'AND',
    'not': 'NOT',
    'in': 'IN',
}

tokens = (
    'LBRACE', 'RBRACE', 'PRINT', 'LPAREN', 'RPAREN', 'SEMI', 'NUMBER', 'PLUS', 'MINUS', 'TIMES', 'DIVIDE',
    'MODULUS', 'FLOORDIV', 'EXPONENT', 'STRING', 'BOOLEAN', 'AND', 'OR', 'NOT', 'IN', 'LT', 'ID', 'ASSIGN', 'IF',
    'ELSE', 'WHILE', 'LTE', 'EQUAL', 'NOTEQUAL', 'GT', 'GTE', 'COMMA', 'LBRACK', 'RBRACK',
)

t_ignore = " \t"

t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_PRINT = 'print'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_ASSIGN = r'='
t_IF = 'if'
t_ELSE = 'else'
t_WHILE = 'while'
t_OR = r'or'
t_AND = r'and'
t_NOT = r'not'
t_IN = r'in'
t_FLOORDIV = r'//'
t_MODULUS = r'%'
t_EXPONENT = r'\*\*'
t_LT = r'<'
t_LTE = r'<='
t_EQUAL = r'=='
t_NOTEQUAL = r'<>'
t_GT = r'>'
t_GTE = r'>='
t_COMMA = r','
t_LBRACK = r'\['
t_RBRACK = r']'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_SEMI = r';'
t_PLUS = r'\+'


def t_NUMBER(t):
    r'-?\d*(\d\.|\.\d)\d* | \d+'
    try:
        t.value = NumberNode(t.value)
    except ValueError:
        print("Integer value too large %d", t.value)
        t.value = 0
    return t


def t_BOOLEAN(t):
    r' True | False '
    if t.value == 'True':
        t.value = BooleanNode(None, True, None)
    else:
        t.value = BooleanNode(None, False, None)
    return t


def t_STRING(t):
    r'("[^"]*")|(\'[^\']*\')'
    t.value = StringNode(t.value[1:len(t.value) - 1])
    return t


def t_ID(t):
    r'[A-Za-z][A-Za-z0-9_]*'
    t.type = reserved.get(t.value, 'ID')
    if t.type == 'ID':
        t.value = VariableNode(t.value)
    return t


def t_error(t):
    print("SYNTAX ERROR")


lex.lex()

precedence = (
    ('left', 'OR'),
    ('left', 'AND'),
    ('left', 'NOT'),
    ('left', 'LT', 'LTE', 'EQUAL', 'NOTEQUAL', 'GT', 'GTE'),
    ('left', 'IN'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'FLOORDIV'),
    ('left', 'MODULUS'),
    ('left', 'TIMES', 'DIVIDE'),
    ('right', 'EXPONENT'),
    ('left', 'LBRACK'),
    ('left', 'LPAREN', 'RPAREN'),
    ('left', 'NUMBER', 'STRING', 'BOOLEAN'),
)

variables = {}


def p_block(t):
    """
    block : LBRACE inblock RBRACE
    """
    t[0] = t[2]


def p_emptyBlock(t):
    """
    block : LBRACE RBRACE
    """
    t[0] = BlockNode([])
    t[0].sl = []


def p_inblock(t):
    """
    inblock : smt inblock
            | block inblock
    """
    t[0] = t[2]
    t[0].sl.insert(0, t[1])


def p_inblock2(t):
    """
    inblock : smt
            | inempty
    """
    t[0] = BlockNode(t[1])


def p_inblock_empty(t):
    """
    inempty :
    """
    pass


def p_smt(t):
    """
    smt : print_smt
        | assign_smt
        | if_smt
        | else_smt
        | while_smt
    """
    t[0] = t[1]


def p_print(t):
    """
    print_smt : PRINT LPAREN expression RPAREN SEMI
    """
    t[0] = PrintNode(t[3])


def p_assign(t):
    """
    assign_smt : ID ASSIGN expression SEMI
    """
    t[0] = AssignNode(t[1].name, t[3], None)


def p_assign_list(t):
    """
    assign_smt : expression LBRACK expression RBRACK ASSIGN expression SEMI
    """
    t[0] = AssignNode(t[1], t[6], t[3])


def p_if(t):
    """
    if_smt : IF LPAREN expression RPAREN block
    """
    t[0] = IfNode(t[3], t[5])


def p_ifelse(t):
    """
    else_smt : if_smt ELSE block
    """
    t[0] = IfElseNode(t[1], t[3])


def p_while(t):
    """
    while_smt : WHILE LPAREN expression RPAREN block
    """
    t[0] = WhileNode(t[3], t[5])


def p_expression_binop(t):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression EXPONENT expression
                  | expression MODULUS expression
                  | expression FLOORDIV expression
                  | expression LT expression
                  | expression LTE expression
                  | expression EQUAL expression
                  | expression NOTEQUAL expression
                  | expression GT expression
                  | expression GTE expression
                  | expression IN expression'''
    t[0] = BopNode(t[2], t[1], t[3])


def p_bool(t):
    '''expression : NOT expression
                  | expression AND expression
                  | expression OR expression'''
    if len(t) == 3:
        t[0] = BooleanNode(t[1], t[2], None)
    elif len(t) == 4:
        t[0] = BooleanNode(t[2], t[1], t[3])


def p_paren(t):
    'expression : LPAREN expression RPAREN'
    t[0] = t[2]


def p_list(t):
    '''
    list : LBRACK inlist RBRACK
         | LBRACK RBRACK
    '''
    if len(t) == 3:
        t[0] = ListNode(None)
    else:
        t[0] = t[2]


def p_inlist(t):
    '''
    inlist : expression
           | expression COMMA inlist
    '''
    if len(t) == 2:
        t[0] = ListNode(t[1])
    else:
        t[0] = t[3]
        t[0].s1.insert(0, t[1])


def p_list_index(t):
    '''
    index : expression LBRACK expression RBRACK
    '''
    t[0] = IndexNode(t[1], t[3])


def p_factor(t):
    '''expression : factor'''
    t[0] = t[1]


def p_factor_number(t):
    'factor : NUMBER'
    t[0] = t[1]


def p_expression_boolean(t):
    'expression : BOOLEAN'
    t[0] = t[1]


def p_expression_list(t):
    'expression : list'
    t[0] = t[1]


def p_expression_index(t):
    'expression : index'
    t[0] = t[1]


def p_expression_string(t):
    'expression : STRING'
    t[0] = t[1]


def p_expression_var(t):
    'expression : ID'
    t[0] = variables.get(t[1].name, t[1])


yacc.yacc()


def main():
    test = open(sys.argv[1], mode='r')

    code = ''
    for line in test:
        toParse = line.strip()
        code += toParse

    try:
        ast = yacc.parse(code)
        ast.execute()
    except SyntaxError:
        print("Syntax Error")
    except Exception:
        print("Semantic Error")


main()
