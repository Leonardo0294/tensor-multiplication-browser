// Función para generar un número aleatorio entre 1 y 'n'
const randomNumber = (n) => Math.floor(Math.random() * (n)) + 1;

// Selección del botón de cálculo por su ID
const calculateButton = document.querySelector("#calculate-btn");

// Función para convertir megabytes a bytes
const megabytesToBytes = (megabytes) => megabytes * 1e+6;

// Función para convertir bytes a megabytes
const bytesToMegabytes = (bytes) => bytes / 1e+6;

// Tamaño de un float32 en bytes.
const FLOAT_32_SIZE = 4;

// Función para calcular el tamaño en bytes de un tensor 2D
const getTensor2dByteSize = (tensor) => {
    const [x, y] = tensor.shape;
    return x * y * (FLOAT_32_SIZE);
}

// Constante que representa 64 megabytes en bytes
const SIXTY_FOUR_MB_BYTES = megabytesToBytes(64);

// Event Listener para el botón de cálculo
calculateButton.addEventListener('click', () => {
    // Realizar operaciones dentro de tf.tidy() para liberar memoria
    tf.tidy(() => {
        // Crear matrices de números aleatorios para los tensores A y B
        const arrA = Array.from({ length: 100 * 100 }).map(() => randomNumber(9));
        const arrB = Array.from({ length: 100 * 100 }).map(() => randomNumber(9));
        
        // Crear tensores 2D a partir de las matrices
        let tensorA = tf.tensor2d(arrA, [100, 100]);
        const tensorB = tf.tensor2d(arrB, [100, 100]);

        // Imprimir los valores de los tensores A y B en la consola
        tensorA.print();
        tensorB.print();
        let tensorQuantity = 2;

        // Realizar bucle mientras el tamaño del tensorA sea menor que 64 MB en bytes
        while (getTensor2dByteSize(tensorA) < SIXTY_FOUR_MB_BYTES) {
            // Concatenar tensorA con tensorB
            const prod = tensorA.concat(tensorB);

            // Liberar memoria del tensorA original y asignar el nuevo tensor concatenado
            tensorA.dispose();
            tensorA = prod;

            // Incrementar la cantidad de tensores
            tensorQuantity += 1;

            // Calcular y mostrar el tamaño del tensorA en bytes y megabytes en la consola
            const bytes = getTensor2dByteSize(tensorA);
            console.log("Tamaño de tensorA: (bytes)", bytes);
            console.log("Tamaño de tensorA: (megabytes)", bytesToMegabytes(bytes));
        }

        // Mostrar el tamaño final de tensorA y la cantidad total de tensores creados
        console.log("Tamaño final de tensorA: ", getTensor2dByteSize(tensorA));
        console.log("Cantidad de tensores: ", tensorQuantity);
    });

    // Mostrar la cantidad de tensores actualmente en memoria en la consola
    console.log("Cantidad de tensores en memoria: ", tf.memory().numTensors);
});
