#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_ROWS 1000
#define MAX_NAME 100
#define MAX_SEX 10
#define MAX_TICKET 20
#define MAX_CABIN 20
#define MAX_DEPTH 10  // Profundidade máxima aumentada


typedef struct {
    int passengerId;
    int survived;
    int pclass;
    char name[MAX_NAME];
    char sex[MAX_SEX];
    float age;
    int sibSp;
    int parch;
    char ticket[MAX_TICKET];
    float fare;
    char cabin[MAX_CABIN];
    char embarked;
} Passenger;

typedef struct TreeNode {
    int featureIndex;
    float threshold;
    int isLeaf;
    int prediction;
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode;

Passenger trainData[MAX_ROWS];
Passenger validationData[MAX_ROWS];
int trainSize = 0, validationSize = 0;

// Função para ler CSV e armazenar no array
int readCSV(const char* filename, Passenger* data) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Erro ao abrir o arquivo %s\n", filename);
        return 0;
    }

    char line[256];
    fgets(line, sizeof(line), file); // Ignorar cabeçalho
    int count = 0;

    while (fgets(line, sizeof(line), file) && count < MAX_ROWS) {
        Passenger p;
        char sexStr[MAX_SEX], cabinStr[MAX_CABIN], embarkedStr[2];

        sscanf(line, "%d,%d,%d,\"%[^\"]\",%[^,],%f,%d,%d,%[^,],%f,%[^,],%s",
               &p.passengerId, &p.survived, &p.pclass, p.name,
               sexStr, &p.age, &p.sibSp, &p.parch, p.ticket,
               &p.fare, cabinStr, embarkedStr);

        strcpy(p.sex, sexStr);
        strcpy(p.cabin, cabinStr);
        p.embarked = embarkedStr[0];

        data[count++] = p;
    }

    fclose(file);
    return count;
}

// Cálculo do índice de Gini
float giniImpurity(Passenger* data, int size, int featureIndex, float threshold) {
    int leftCount = 0, rightCount = 0;
    int leftSurvived = 0, rightSurvived = 0;

    for (int i = 0; i < size; i++) {
        if (((featureIndex == 5) && (data[i].age <= threshold)) ||  // Idade
            ((featureIndex == 9) && (data[i].fare <= threshold))) { // Tarifa
            leftCount++;
            if (data[i].survived == 1) leftSurvived++;
        } else {
            rightCount++;
            if (data[i].survived == 1) rightSurvived++;
        }
    }

    float leftProb = leftCount ? (float)leftSurvived / leftCount : 0;
    float rightProb = rightCount ? (float)rightSurvived / rightCount : 0;

    float giniLeft = 1 - (leftProb * leftProb + (1 - leftProb) * (1 - leftProb));
    float giniRight = 1 - (rightProb * rightProb + (1 - rightProb) * (1 - rightProb));

    return (leftCount * giniLeft + rightCount * giniRight) / size;
}

// Encontra o melhor ponto de divisão
void bestSplit(Passenger* data, int size, int* bestFeature, float* bestThreshold) {
    float bestGini = 1.0;
    *bestFeature = -1;
    *bestThreshold = 0;

    for (int feature = 5; feature <= 9; feature += 4) { // Idade (5) e Tarifa (9)
        for (int i = 0; i < size; i++) {
            float threshold = (feature == 5) ? data[i].age : data[i].fare;
            float gini = giniImpurity(data, size, feature, threshold);
            if (gini < bestGini) {
                bestGini = gini;
                *bestFeature = feature;
                *bestThreshold = threshold;
            }
        }
    }
    printf("Melhor divisão: Feature %d, Threshold %.2f\n", *bestFeature, *bestThreshold);
}

// Construção da Árvore de Decisão
TreeNode* buildTree(Passenger* data, int size, int depth) {
    if (size == 0 || depth >= MAX_DEPTH) return NULL;

    int survivedCount = 0;
    for (int i = 0; i < size; i++) survivedCount += data[i].survived;
    if (survivedCount == 0 || survivedCount == size) {
        TreeNode* leaf = (TreeNode*)malloc(sizeof(TreeNode));
        leaf->isLeaf = 1;
        leaf->prediction = (survivedCount > size / 2) ? 1 : 0;
        return leaf;
    }

    int bestFeature;
    float bestThreshold;
    bestSplit(data, size, &bestFeature, &bestThreshold);

    if (bestFeature == -1) {
        TreeNode* leaf = (TreeNode*)malloc(sizeof(TreeNode));
        leaf->isLeaf = 1;
        leaf->prediction = (survivedCount > size / 2) ? 1 : 0;
        return leaf;
    }

    Passenger* leftData = (Passenger*)malloc(size * sizeof(Passenger));
    Passenger* rightData = (Passenger*)malloc(size * sizeof(Passenger));
    int leftSize = 0, rightSize = 0;

    for (int i = 0; i < size; i++) {
        if ((bestFeature == 5 && data[i].age <= bestThreshold) || 
            (bestFeature == 9 && data[i].fare <= bestThreshold)) {
            leftData[leftSize++] = data[i];
        } else {
            rightData[rightSize++] = data[i];
        }
    }

    TreeNode* node = (TreeNode*)malloc(sizeof(TreeNode));
    node->isLeaf = 0;
    node->featureIndex = bestFeature;
    node->threshold = bestThreshold;
    node->left = buildTree(leftData, leftSize, depth + 1);
    node->right = buildTree(rightData, rightSize, depth + 1);

    free(leftData);
    free(rightData);
    return node;
}

// Predição com a árvore
int predict(TreeNode* tree, Passenger p) {
    if (tree->isLeaf) return tree->prediction;

    if ((tree->featureIndex == 5 && p.age <= tree->threshold) || 
        (tree->featureIndex == 9 && p.fare <= tree->threshold)) {
        return predict(tree->left, p);
    } else {
        return predict(tree->right, p);
    }
}

// Função recursiva para imprimir a árvore de decisão
void printTree(TreeNode* node, int level) {
    if (node == NULL) {
        return;
    }

    // Indenta a árvore para mostrar a profundidade (nível)
    for (int i = 0; i < level; i++) {
        printf("\t");
    }

    // Se for um nó folha, imprime a predição
    if (node->isLeaf) {
        printf("Leaf: Predição = %d\n", node->prediction);
    } else {
        // Caso contrário, imprime a divisão e o valor do threshold
        printf("Feature %d ", node->featureIndex);
        if (node->featureIndex == 5) {
            printf("(Age) ");
        } else if (node->featureIndex == 9) {
            printf("(Fare) ");
        }
        printf("Threshold = %.2f\n", node->threshold);

        // Chama recursivamente para os filhos esquerdo e direito
        printf("Left -> ");
        printTree(node->left, level + 1);

        printf("Right -> ");
        printTree(node->right, level + 1);
    }
}


int main() {
    trainSize = readCSV("train.csv", trainData);
    validationSize = readCSV("validation.csv", validationData);

    printf("Treinando árvore de decisão...\n");
    TreeNode* decisionTree = buildTree(trainData, trainSize, 0);

    // Exibe a árvore de decisão
    printf("Árvore de Decisão:\n");
    printTree(decisionTree, 0);  // A partir do nó raiz com nível 0

    printf("Fazendo predições...\n");
    for (int i = 0; i < validationSize; i++) {
        int result = predict(decisionTree, validationData[i]);
        printf("Passageiro %d -> Predição: %d\n", validationData[i].passengerId, result);
    }

    return 0;
}
