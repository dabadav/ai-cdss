document.addEventListener("DOMContentLoaded", function () {
    const modal = document.createElement("div");
    modal.classList.add("modal");
    modal.id = "img-modal";
    modal.innerHTML = `
      <span class="modal-close">&times;</span>
      <img class="modal-content" id="modal-img">
    `;
    document.body.appendChild(modal);
  
    const modalImg = document.getElementById("modal-img");
    const closeBtn = modal.querySelector(".modal-close");
  
    document.querySelectorAll("img.zoomable").forEach((img) => {
      img.style.cursor = "zoom-in";
      img.addEventListener("click", () => {
        modal.style.display = "block";
        modalImg.src = img.src;
        modalImg.alt = img.alt;
      });
    });
  
    closeBtn.onclick = function () {
      modal.style.display = "none";
    };
  
    modal.onclick = function (event) {
      if (event.target === modal) {
        modal.style.display = "none";
      }
    };
  });
  