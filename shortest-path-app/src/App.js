import React, { useState, useEffect } from 'react';
import MapComponent from './MapComponent';
import './App.css';

function App() {
    const [startNode, setStartNode] = useState(null);
    const [endNode, setEndNode] = useState(null);
    const [destinationLat, setDestinationLat] = useState('');
    const [destinationLng, setDestinationLng] = useState('');

    useEffect(() => {
        // Function to get current location
        const getCurrentLocation = () => {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        setStartNode({
                            id: 'Current Location',
                            coordinates: [position.coords.latitude, position.coords.longitude],
                        });
                    },
                    (error) => {
                        alert(error.message);
                    },
                    {
                        enableHighAccuracy: true,
                        timeout: 20000,
                        maximumAge: 1000,
                    }
                );
            } else {
                alert('Geolocation is not supported by this browser.');
            }
        };

        // Get location on component mount
        getCurrentLocation();
    }, []);

    const handleDestinationSelection = (coordinates) => {
        setEndNode({
            id: 'Selected Destination',
            coordinates: coordinates,
        });
    };

    const handleManualDestination = () => {
        // Validate inputs
        if (destinationLat === '' || destinationLng === '') {
            alert('Please enter both latitude and longitude.');
            return;
        }

        // Parse latitude and longitude
        const lat = parseFloat(destinationLat);
        const lng = parseFloat(destinationLng);

        // Validate parsed values
        if (isNaN(lat) || isNaN(lng)) {
            alert('Please enter valid numeric values for latitude and longitude.');
            return;
        }

        setEndNode({
            id: 'Manual Destination',
            coordinates: [lat, lng],
        });
    };

    return (
        <div className="app-container">
            <h1>Indian City Route Finder</h1>
            <div className="input-section">
                <label htmlFor="destination-lat">Destination Latitude:</label>
                <input
                    type="text"
                    id="destination-lat"
                    value={destinationLat}
                    onChange={(e) => setDestinationLat(e.target.value)}
                />
                <label htmlFor="destination-lng">Destination Longitude:</label>
                <input
                    type="text"
                    id="destination-lng"
                    value={destinationLng}
                    onChange={(e) => setDestinationLng(e.target.value)}
                />
                <button onClick={handleManualDestination}>Find Directions</button>
            </div>
            {startNode ? (
                <MapComponent
                    startNode={startNode}
                    endNode={endNode}
                    onDestinationSelect={handleDestinationSelection}
                />
            ) : (
                <p>Loading map...</p>
            )}
        </div>
    );
}

export default App;

