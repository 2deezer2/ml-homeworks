import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.
     
    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    n = len(target_vector)

    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_classes = target_vector[sorted_indices]

    unique_features, indices = np.unique(sorted_features[::-1], return_index = True)
    true_indices = n - 1 - indices #Разворачиваем, сортированный массив, чтобы были последние вхождения уникальных признаков, так как именно они и являются разделяющими
    total = np.sum(target_vector)
    #Сначала думал брать разбиения, считая пороги до индексов, но не получилось
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_classes = target_vector[sorted_indices]


    unique_thresholds = (unique_features[:-1] + unique_features[1:]) / 2

    left_sum = np.cumsum(sorted_classes)[true_indices[:-1]]
    left_sizes = np.arange(n)[true_indices[:-1]] + 1

    #обрабатываем исключение относительно размера разбиения, чтобы не было пустого множества.
    if true_indices[-1] != (n - 1):
        left_sum = np.cumsum(sorted_classes)[true_indices] 
        left_sizes = np.arange(n)[true_indices] + 1

    right_sum = total - left_sum
    right_sizes = n - left_sizes

    q_r_l = left_sum * (left_sizes - left_sum) / left_sizes #Решил взять формулу с картинки

    q_r_r = right_sum * (right_sizes - right_sum) / right_sizes

    Q = -2 * (q_r_l + q_r_r) / n

    best_index = np.argmax(Q)
    gini_best = Q[best_index]
    threshold_best = unique_thresholds[best_index]

    return unique_thresholds, Q, threshold_best, gini_best

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth = 1): #Дополнил глубиной, чтобы можно было трекать критерий останова.
        if np.all(sub_y == sub_y[0]): #Поменял неравенство на ==, чтобы был правильный критерий
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if self._max_depth is not None:
            if depth >= self._max_depth: #Проверка на критерий останова
                node["type"] = "terminal"
                node["class"] = sub_y[0]
                return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]): #Тут с 1 начиналось LUL
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count #Поменял числитель и знаменатель
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))) #Тут опять нужно менять 1 на нолик в первой лямбда-функции
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature]))) #Тут возвращался map
            else:
                raise ValueError

            if len(feature_vector) == 1:#Исключение по одному признаку
                continue

            if np.all(feature_vector == feature_vector[0]): #Один признак значит больше нельзя разбить
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical": #Описка
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] #Тут сделал [0][0], потому что возвращался список из 1 кортежа
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)#Здесь дописал текущую глубину
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        if node['type'] == 'terminal':
            return node['class'] #Если лист конечный, тогда сразу возвращаем класс
        
        if self._feature_types[node['feature_split']] == 'real': #Иначе делим на предсказание подлистов.
            if x[node['feature_split']] > node['threshold']: 
                return self._predict_node(x, node['right_child'])
            else:
                return self._predict_node(x, node['left_child'])
        else:
            if x[node['feature_split']] in node['categories_split']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
