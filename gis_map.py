import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.set_page_config(
    page_title="RoadSafe AI GIS",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ RoadSafe AI — GIS Hotspot Map")

st.warning(
    "DEMO MAP: Grid cells are illustrative and do not represent "
    "verified accident locations from the dataset."
)

st.write(
    "The actual accident grid will be generated after we verify "
    "the highway routes and map their chainage values to GPS coordinates."
)

# Illustrative view centered around Nagpur.
# These coordinates are for the map view only, not accident records.
center_lat = 21.1458
center_lon = 79.0882

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10,
    tiles="OpenStreetMap"
)

# Demonstration grid cells only.
# Do not interpret these cells as real accident hotspots.
demo_cells = [
    (21.1458, 79.0882, "Illustrative high-count cell", "red"),
    (21.1550, 79.1050, "Illustrative medium-count cell", "orange"),
    (21.1300, 79.0700, "Illustrative low-count cell", "yellow"),
]

for lat, lon, label, color in demo_cells:
    folium.Rectangle(
        bounds=[
            [lat - 0.008, lon - 0.008],
            [lat + 0.008, lon + 0.008]
        ],
        color=color,
        fill=True,
        fill_opacity=0.35,
        tooltip=label
    ).add_to(m)

    folium.Marker(
        [lat, lon],
        tooltip=label,
        popup=(
            f"{label}<br>"
            "Illustration only — not a real accident count."
        )
    ).add_to(m)

st_folium(
    m,
    use_container_width=True,
    height=600,
    returned_objects=[]
)

st.subheader("Grid legend")
c1, c2, c3 = st.columns(3)
c1.markdown("🟥 **Red:** Illustrative high concentration")
c2.markdown("🟧 **Orange:** Illustrative medium concentration")
c3.markdown("🟨 **Yellow:** Illustrative low concentration")

st.info(
    "Next: replace the illustrative cells with verified coordinates "
    "and actual accident counts from the dataset."
)

