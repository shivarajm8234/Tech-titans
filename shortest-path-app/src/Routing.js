import React, { useEffect, useState } from 'react';
import { useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet-routing-machine';

function Routing({ startNode, endNode }) {
    const map = useMap();
    const [routingControl, setRoutingControl] = useState(null);
    const [isPanelVisible, setIsPanelVisible] = useState(true); // State to toggle visibility of the panel
    const [routeDetails, setRouteDetails] = useState([]); // Store route instructions

    useEffect(() => {
        if (!map || !startNode || !endNode) return;

        // Initialize routing control
        const control = L.Routing.control({
            waypoints: [
                L.latLng(startNode.coordinates[0], startNode.coordinates[1]),
                L.latLng(endNode.coordinates[0], endNode.coordinates[1]),
            ],
            routeWhileDragging: true,
            show: false,
        }).addTo(map);

        control.on('routesfound', (e) => {
            const routes = e.routes[0].instructions; // Get route instructions
            const instructions = routes.map((step) => step.text); // Extract text instructions
            setRouteDetails(instructions); // Save instructions to state
        });

        setRoutingControl(control);

        return () => {
            map.removeControl(control); // Cleanup on unmount
        };
    }, [startNode, endNode, map]);

    const togglePanelVisibility = () => {
        setIsPanelVisible(!isPanelVisible);
    };

    return (
        <>
            {/* Panel for displaying route details */}
            <div
                style={{
                    position: 'absolute',
                    top: '0',
                    left: isPanelVisible ? '0' : '-300px', // Slide out when hidden
                    width: '300px',
                    height: '100%',
                    backgroundColor: '#fff',
                    boxShadow: '2px 0 5px rgba(0, 0, 0, 0.2)',
                    overflowY: 'auto',
                    transition: 'left 0.3s ease-in-out',
                    zIndex: 1000,
                }}
            >
                <button
                    onClick={togglePanelVisibility}
                    style={{
                        position: 'absolute',
                        top: '10px',
                        right: '-40px',
                        width: '30px',
                        height: '30px',
                        borderRadius: '50%',
                        backgroundColor: '#007bff',
                        color: '#fff',
                        border: 'none',
                        cursor: 'pointer',
                        boxShadow: '0 2px 5px rgba(0, 0, 0, 0.2)',
                    }}
                >
                    {isPanelVisible ? '<' : '>'}
                </button>
                <h3 style={{ padding: '10px', margin: 0 }}>Route Details</h3>
                <ul style={{ paddingLeft: '20px' }}>
                    {routeDetails.length > 0 ? (
                        routeDetails.map((instruction, index) => (
                            <li key={index} style={{ marginBottom: '8px' }}>
                                {instruction}
                            </li>
                        ))
                    ) : (
                        <p style={{ paddingLeft: '10px' }}>No route found.</p>
                    )}
                </ul>
            </div>
        </>
    );
}

export default Routing;

