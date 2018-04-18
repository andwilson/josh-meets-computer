import React from "react";
import Link from "gatsby-link";
import styled from "styled-components";
import Img from "gatsby-image";

export default ({ title, description }) => (
  <div>
    <h1>{title}</h1>
    <p>{description}</p>
  </div>
);
