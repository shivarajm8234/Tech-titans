class PriorityQueue {
    constructor() {
        this.elements = [];
    }

    enqueue(element, priority) {
        this.elements.push({ element, priority });
        this.elements.sort((a, b) => a.priority - b.priority);
    }

    dequeue() {
        return this.elements.shift();
    }

    isEmpty() {
        return this.elements.length === 0;
    }
}

export function dijkstra(graph, startNode, endNode) {
    const distances = {};
    const previous = {};
    const pq = new PriorityQueue();

    // Initialize distances and previous nodes
    for (const node in graph) {
        distances[node] = Infinity;
        previous[node] = null;
    }
    distances[startNode] = 0;

    pq.enqueue(startNode, 0);

    while (!pq.isEmpty()) {
        const { element: currentNode } = pq.dequeue();

        if (currentNode === endNode) break;

        for (const neighbor in graph[currentNode]) {
            const distance = graph[currentNode][neighbor];
            const newDistance = distances[currentNode] + distance;

            if (newDistance < distances[neighbor]) {
                distances[neighbor] = newDistance;
                previous[neighbor] = currentNode;
                pq.enqueue(neighbor, newDistance);
            }
        }
    }

    // Reconstruct the shortest path
    const path = [];
    let current = endNode;
    while (current) {
        path.unshift(current);
        current = previous[current];
    }

    return { path, distance: distances[endNode] };
}
