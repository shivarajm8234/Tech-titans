import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from 'react-leaflet';
import Routing from './Routing';
import 'leaflet/dist/leaflet.css';
import 'leaflet-routing-machine/dist/leaflet-routing-machine.css';
import L from 'leaflet';

// Import custom icons
import currentLocationIconUrl from './icons/current-location.png';
import destinationIconUrl from './icons/destination-marker.png';

// Define custom icons
const currentLocationIcon = new L.Icon({
    iconUrl: currentLocationIconUrl,
    iconSize: [25, 41], // Adjust size as needed
});

const destinationIcon = new L.Icon({
    iconUrl: destinationIconUrl,
    iconSize: [25, 41], // Adjust size as needed
});

function MapComponent({ startNode, endNode, onDestinationSelect }) {
    const [destinationMarker, setDestinationMarker] = useState(null);
    const [userLocation, setUserLocation] = useState(null);
    const mapCenter = [20.5937, 78.9629];

    useEffect(() => {
        if (startNode) {
            setUserLocation([startNode.coordinates[0], startNode.coordinates[1]]);
        }
        if (endNode) {
            setDestinationMarker([endNode.coordinates[0], endNode.coordinates[1]]);
        }
    }, [startNode, endNode]);

    function MapEvents() {
        const map = useMapEvents({
            click: (e) => {
                const { lat, lng } = e.latlng;
                setDestinationMarker([lat, lng]);
                onDestinationSelect([lat, lng]);
            },
        });
        return null;
    }

    return (
        <div className="map-container">
            <MapContainer center={mapCenter} zoom={5} style={{ height: "600px", width: "800px" }}>
                <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                />
                {userLocation && (
                    <Marker position={userLocation} icon={currentLocationIcon}>
                        <Popup>Your Current Location</Popup>
                    </Marker>
                )}
                {destinationMarker && (
                    <Marker position={destinationMarker} icon={destinationIcon}>
                        <Popup>Selected Destination</Popup>
                    </Marker>
                )}
                {startNode && endNode && <Routing startNode={startNode} endNode={endNode} />}
                <MapEvents />
            </MapContainer>
        </div>
    );
}

export default MapComponent;

