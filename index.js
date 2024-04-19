// Definir una matriz vacía
const matrix = [];

// Crear una matriz 10x10 con números del 1 al 10
for (let i = 0; i < 10; i++) {
    let row = [];
    for (let j = 0; j < 10; j++) {
        // Agregar números del 1 al 10 a cada fila de la matriz
        if (j + 1 === 10) {
            row.push(1); // Si j+1 es igual a 10, agregar 1 en lugar de 10
        } else {
            row.push(j + 1); // Agregar el número actual (j + 1)
        }
    }
    matrix.push(row); // Agregar la fila a la matriz
}

// Función asincrónica para verificar el uso de memoria
async function memoryVerify() {
    const bytesMaxSize = 64 * 1024 * 1024; // Tamaño máximo de memoria en bytes (64 MB)

    // Inicializar tensores con la matriz creada
    let tensor1 = tf.tensor2d(matrix);
    let tensor2 = tf.tensor2d(matrix);
    let finalTensor = tf.tensor2d(matrix);

    let firstTensor = true; // Variable para alternar entre tensores en cada iteración

    console.log('Inicio de la verificación de memoria...'); // Mensaje de inicio

    // Bucle infinito para realizar operaciones y monitorear la memoria
    while (true) {
        if (firstTensor) {
            // En la primera iteración, multiplicar el tensor final por tensor1 y actualizar tensor1
            finalTensor = finalTensor.mul(tensor1);
            tensor1 = tensor1.mul(tensor2);
        } else {
            // En las iteraciones subsiguientes, multiplicar el tensor final por tensor2 y actualizar tensor2
            finalTensor = finalTensor.mul(tensor2);
            tensor2 = tensor1.mul(tensor1);
        }

        // Obtener el estado de la memoria actual
        const tensorMemory = tf.memory();
        const memoryInMB = tensorMemory.numBytes / (1024 * 1024); // Convertir bytes a megabytes
        const consoleMsg = `Uso de memoria: ${memoryInMB.toFixed(2)} MB`; // Crear mensaje de uso de memoria

        console.log(consoleMsg); // Mostrar mensaje de uso de memoria en la consola

        // Verificar si se superó el límite máximo de memoria
        if (tensorMemory.numBytes > bytesMaxSize) {
            console.log('Límite de 64MB alcanzado.'); // Mostrar mensaje de límite alcanzado
            console.log('Tensor final:');
            console.log(finalTensor.toString()); // Mostrar el tensor final como cadena
            finalTensor.print(); // Imprimir el tensor final en la consola

            // Liberar memoria al eliminar los tensores
            tensor1.dispose();
            tensor2.dispose();
            finalTensor.dispose();

            break; // Salir del bucle
        }

        firstTensor = !firstTensor; // Alternar entre tensores en cada iteración

        await new Promise(resolve => setTimeout(resolve, 1)); // Esperar 1 milisegundo antes de la siguiente iteración
    }
}

memoryVerify(); // Llamar a la función para iniciar la verificación de memoria
