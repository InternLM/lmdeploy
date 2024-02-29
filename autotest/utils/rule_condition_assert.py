def assert_result(input, rule_condition, model_name):
    input = input.replace('\n', '\\n')
    input_lower = input.lower()
    for dict in rule_condition:
        if dict is None:
            return True, ''

        for rule in dict:
            operator = list(rule.keys())[0]
            value = list(rule.values())[0]
            if model_name is not None and model_name == operator:
                dict = value

        for rule in dict:
            operator = list(rule.keys())[0]
            value = list(rule.values())[0]
            if operator == 'contain':
                if isinstance(value, list):
                    tmpResult = False
                    for word in value:
                        if word.lower() in input_lower:
                            tmpResult = True
                    if tmpResult is False:
                        return False, ','.join(
                            value) + " doesn't exist in " + input
                else:
                    if value.lower() not in input_lower:
                        msg = value + " doesn't exist in:" + input
                        return False, msg
            if operator == 'not_contain':
                if isinstance(value, list):
                    for word in value:
                        if word.lower() in input_lower:
                            msg = word + " shouldn't exist in:" + input
                            return False, msg
                else:
                    if value.lower() in input_lower:
                        msg = value + " shouldn't exist in " + input
                        return False, msg
            if operator == 'len_g':
                if len(input) < int(value):
                    return False, input + ' length: ' + str(
                        len(input)) + ', should greater than ' + str(value)
        return True, ''


if __name__ == '__main__':
    input = '成都的景点hot potdddd'
    condition = ([[{
        'contain': ['hot pot']
    }, {
        'contain': ['。']
    }, {
        'len_g': [10]
    }]])
    print(assert_result(input, condition))
